"""Fused all-reduce + residual-add + RMSNorm via FlashInfer.

A tensor-parallel transformer layer ends every parallel region the same way:
all-reduce the partial output, add the residual, normalise. Run as three steps
that is one collective kernel plus a norm kernel, and the activation crosses HBM
twice. FlashInfer's ``allreduce_fusion`` does all three in a single kernel with
one round trip, launched with PDL so it starts while its predecessor drains.

This is what vLLM's ``allreduce_rms`` inductor fusion lowers to
(``vllm/compilation/passes/fusion/allreduce_rms_fusion.py``), and it is worth
0.39 ms/step of gpt-oss-120b decode at tp=2 on B200 -- measured on vLLM itself
by toggling ``pass_config.fuse_allreduce_rms`` (2.684 -> 2.292 ms/step).

The kernel needs a preallocated workspace sized for a maximum token count, so it
cannot serve arbitrarily large batches. Above that bound
:func:`fused_allreduce_add_rmsnorm` falls back to the unfused pair, which is what
vLLM does too -- it just decides at compile time (``is_applicable_for_range``)
where we decide at runtime, because we compile one dynamic graph rather than one
per token range.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_FI_AVAILABLE: bool | None = None
_flashinfer_comm = None
_TorchDistBackend = None

# Workspaces are keyed by everything that changes their layout. In practice a
# run creates exactly one, but keying avoids silently reusing a workspace sized
# for a different hidden dim (which would corrupt rather than fail).
_WORKSPACES: dict[tuple[int, int, torch.dtype, int], object] = {}

# Zero residual for the no-residual pattern, one per (hidden, dtype), sized for
# the workspace's whole token range and sliced per call.
#
# This MUST be allocated outside any CUDA graph capture. Allocating it lazily on
# first use placed it in the capturing graph's private memory pool; reusing the
# cached tensor from another graph or from eager is then undefined behaviour, and
# it silently corrupted memory and killed a rank mid-capture with no traceback.
# It is created alongside the workspace instead, which happens during the eager
# warmup forward, well before decode capture.
_ZERO_RESIDUALS: dict[tuple[int, torch.dtype], torch.Tensor] = {}

# Max fused-collective payload per (device capability, world size), in MB.
# Copied from vLLM's ``FI_ALLREDUCE_FUSION_MAX_SIZE_MB`` -- these are FlashInfer
# workspace limits, not tuning knobs, so they have to agree with vLLM's.
_FI_MAX_SIZE_MB: dict[int, dict[int, float]] = {
    90: {2: 64, 4: 2, 8: 0.5},
    100: {2: 64, 4: 32, 8: 1, 16: 64},
    103: {2: 64, 4: 64, 8: 2, 16: 64},
}

# Below this token count FlashInfer's one-shot path signals PDL completion
# early; vLLM's ``PDL_ADVANCE_LAUNCH_TOKENS``.
_PDL_ADVANCE_LAUNCH_TOKENS = 16

_MiB = 1024 * 1024

# Upper bound on tokens we will size a workspace for, mirroring the
# ``min(size_cap, max_num_batched_tokens)`` clamp vLLM applies to avoid
# allocating a workspace larger than any batch can use.
_MAX_BATCHED_TOKENS = 16384


import flashinfer.comm as flashinfer_comm
from flashinfer.comm.mnnvl import TorchDistBackend


def fi_ar_fusion_available() -> bool:
    """True when FlashInfer exposes ``allreduce_fusion`` and it is not disabled."""
    global _FI_AVAILABLE, _flashinfer_comm, _TorchDistBackend
    if _FI_AVAILABLE is not None:
        return _FI_AVAILABLE
    if os.environ.get("FASTKERNELS_FI_ALLREDUCE_FUSION", "1") == "0":
        _FI_AVAILABLE = False
        return False
    _FI_AVAILABLE = hasattr(flashinfer_comm, "allreduce_fusion")
    _flashinfer_comm = flashinfer_comm
    _TorchDistBackend = TorchDistBackend
    return _FI_AVAILABLE


def fi_ar_fusion_max_token_num(hidden_dim: int, dtype: torch.dtype,
                               world_size: int) -> int | None:
    """Largest token count the fused kernel can serve, or None if unsupported.

    None means this (capability, world size) pair has no FlashInfer workspace
    configuration at all, so the fusion must stay off entirely.
    """
    if world_size not in (2, 4, 8, 16):
        return None
    capability = torch.cuda.get_device_capability()
    cap_int = capability[0] * 10 + capability[1]
    max_size_mb = _FI_MAX_SIZE_MB.get(cap_int, {}).get(world_size)
    if max_size_mb is None:
        return None
    element_size = torch.tensor([], dtype=dtype).element_size()
    return min(int(max_size_mb * _MiB) // (hidden_dim * element_size),
               _MAX_BATCHED_TOKENS)


def get_fi_ar_workspace(hidden_dim: int, dtype: torch.dtype):
    """Return (workspace, max_token_num), creating the workspace on first use.

    Returns ``(None, 0)`` when the fusion is unavailable on this topology --
    mnnvl needs NVSwitch multicast, so workspace creation is a real runtime
    check, not a formality.
    """
    if not fi_ar_fusion_available() or not dist.is_initialized():
        return None, 0
    world_size = dist.get_world_size()
    if world_size <= 1:
        return None, 0
    max_token_num = fi_ar_fusion_max_token_num(hidden_dim, dtype, world_size)
    if max_token_num is None or max_token_num < 1:
        return None, 0

    rank = dist.get_rank()
    key = (world_size, hidden_dim, dtype, max_token_num)
    if key in _WORKSPACES:
        return _WORKSPACES[key], max_token_num

    try:
        workspace = _flashinfer_comm.create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=_TorchDistBackend(group=None),
            group=None,
        )
    except Exception as exc:                                    # noqa: BLE001
        # Expected on topologies without NVSwitch multicast. vLLM falls back to
        # the ``trtllm`` backend there; try the same before giving up.
        if not torch.compiler.is_compiling():
            # Dynamo rejects logging.Logger methods under fullgraph=True, and
            # this lazy setup is reachable from a traced forward.
            logger.warning(
                "FlashInfer mnnvl allreduce workspace unavailable (%s); "
                "trying trtllm backend", exc,
            )
        try:
            workspace = _flashinfer_comm.create_allreduce_fusion_workspace(
                backend="trtllm",
                world_size=world_size,
                rank=rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                comm_backend=_TorchDistBackend(group=None),
                group=None,
            )
        except Exception as exc2:                               # noqa: BLE001
            if not torch.compiler.is_compiling():
                logger.warning(
                    "FlashInfer allreduce fusion disabled: no usable workspace "
                    "(%s)", exc2,
                )
            _WORKSPACES[key] = None
            return None, 0

    # Allocate the zero residual here, where we are certainly not capturing.
    if (hidden_dim, dtype) not in _ZERO_RESIDUALS:
        _ZERO_RESIDUALS[(hidden_dim, dtype)] = torch.zeros(
            max_token_num, hidden_dim, dtype=dtype, device="cuda",
        )

    if not torch.compiler.is_compiling():
        logger.info(
            "Initialized FlashInfer allreduce fusion workspace: backend=%s "
            "world_size=%d hidden_dim=%d max_token_num=%d dtype=%s",
            getattr(workspace, "backend", "?"), world_size, hidden_dim,
            max_token_num, dtype,
        )
    _WORKSPACES[key] = workspace
    return workspace, max_token_num


def destroy_fi_ar_workspaces() -> None:
    """Release every workspace. Safe to call when none were created."""
    for key, workspace in list(_WORKSPACES.items()):
        if workspace is not None:
            try:
                workspace.destroy()
            except Exception:                                   # noqa: BLE001
                pass
        _WORKSPACES.pop(key, None)


def _unfused(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
             eps: float, weight_bias: float = 0.0,
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce then fused_add_rms_norm -- the path this op replaces.

    Used above the workspace's token bound and whenever the fusion is
    unavailable. Goes through the same two kernels the unfused graph would, so
    falling back costs nothing beyond not being fused.

    ``weight_bias=1.0`` selects the Gemma convention, where the checkpoint
    stores the scale as an offset from 1.0 and the norm must apply
    ``(1 + weight)`` in fp32.
    """
    out = torch.ops.fastkernels.custom_all_reduce(x)
    if weight_bias == 1.0:
        from ..L1.gemma_rms_norm import GemmaRMSNorm

        return GemmaRMSNorm._forward_static_with_residual(
            weight, eps, out, residual,
        )

    from ..L1.rms_norm import RMSNorm

    res = residual.clone()
    return RMSNorm.forward_cuda(out, weight, eps, res)


def fused_allreduce_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``all_reduce(x) + residual``, RMSNormed, in one kernel.

    ``weight_bias`` is added to ``rms_gamma`` inside the kernel, in fp32. Pass
    1.0 for Gemma-convention norms (Qwen3-Next, Gemma) so the ``(1 + weight)``
    scale is formed at full precision rather than rounded into bf16 by the
    caller -- the same knob vLLM's ``AllReduceFusedAddGemmaRMSNormPattern``
    uses.

    Returns ``(normed, new_residual)`` where ``new_residual`` is the summed
    activation before normalisation -- the same two values the unfused
    ``all_reduce`` + ``fused_add_rms_norm`` pair produces, to within bf16
    rounding of the norm (the reduction itself is bitwise identical).

    Functional by construction: outputs are fresh buffers and the inputs are
    left untouched, so Inductor needs no auto-functionalization to place it.
    """
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    x2d = x.reshape(-1, hidden_dim)
    res2d = residual.reshape(-1, hidden_dim)
    num_tokens = x2d.shape[0]

    workspace, max_token_num = get_fi_ar_workspace(hidden_dim, x.dtype)
    if workspace is None or num_tokens > max_token_num:
        return _unfused(x, residual, weight, eps, weight_bias)

    x2d = x2d.contiguous()
    res2d = res2d.contiguous()
    norm_out = torch.empty_like(x2d)
    residual_out = torch.empty_like(res2d)

    _flashinfer_comm.allreduce_fusion(
        input=x2d,
        workspace=workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        output=None,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=res2d,
        rms_gamma=weight,
        rms_eps=eps,
        weight_bias=weight_bias,
        # mnnvl sizes its own one-shot workspace around FlashInfer's AUTO
        # strategy, so forcing a choice can request one-shot for a tensor that
        # does not fit it. Let FlashInfer decide (vLLM does the same).
        use_oneshot=None,
        fp32_acc=True,
        # The one-shot Lamport all-reduce signals PDL completion before its
        # output buffer is committed, so a PDL-launched successor can read an
        # uninitialized buffer and produce NaN. vLLM guards this with
        # ``(use_oneshot is True) or num_tokens > PDL_ADVANCE_LAUNCH_TOKENS``,
        # which is safe for them because their pass resolves ``use_oneshot`` per
        # compile range and pins it True on the small-token (decode) range. We
        # leave the choice to FlashInfer's AUTO strategy -- forcing one-shot can
        # request a payload mnnvl's one-shot workspace cannot hold -- so we
        # cannot know here whether one-shot was selected, and the only safe
        # translation of vLLM's condition is to always complete at the end. The
        # early signal is only worth anything for the two-shot path at large
        # token counts, where the collective dwarfs the PDL overlap.
        trigger_completion_at_end=True,
    )
    return norm_out.view(orig_shape), residual_out.view(residual.shape)


def fused_allreduce_add_gemma_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm(all_reduce(x), residual)`` for a Gemma-convention norm, fused.

    ``x`` is the *un-reduced* per-rank partial that a ``RowParallelLinear`` (or
    an MoE's summed expert output) produced with ``reduce_results=False``;
    ``norm`` is the ``GemmaRMSNorm`` that follows it. The Gemma ``(1 + weight)``
    scale is formed inside the kernel in fp32 via ``weight_bias=1.0``, so this
    is not a precision trade against the unfused pair.

    Above the workspace's token bound, or wherever FlashInfer's fused collective
    is unavailable, this falls back to an explicit all-reduce plus ``norm`` --
    the same two kernels, and the same numerics, as leaving the linear reducing.

    This is the eager counterpart of what ``AllReduceFusedAddRMSNormPass``
    rewrites in a compiled graph, and mirrors vLLM's own
    ``fused_allreduce_gemma_rms_norm`` helper for models that run eager.
    """
    hidden_dim = x.shape[-1]
    workspace, max_token_num = get_fi_ar_workspace(hidden_dim, x.dtype)
    if workspace is not None and x.reshape(-1, hidden_dim).shape[0] <= max_token_num:
        return fused_allreduce_add_rmsnorm(
            x, residual, norm.weight, norm.variance_epsilon, weight_bias=1.0,
        )
    return norm(torch.ops.fastkernels.custom_all_reduce(x), residual)


def fused_allreduce_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``all_reduce(x)`` then RMSNorm with no residual, in one kernel.

    This is layer 0's ``input_layernorm``: its input is the vocab-parallel
    embedding's all-reduce and there is no residual yet. One site per step, but it
    is the *first* collective inside the decode CUDA graph, so it absorbs whatever
    inter-rank launch skew the step begins with -- 119.73 us average as an unfused
    ``cross_device_reduce_1stage`` against 5.86 us median for the fused kernel.
    vLLM fuses it too (``AllReduceRMSNormPattern``).

    FlashInfer has no all-reduce + rms_norm pattern without a residual, so a
    zeroed residual is supplied, as vLLM does. With ``residual_in`` zero,
    ``residual_out`` is exactly ``all_reduce(x)`` -- which the caller still needs,
    because at layer 0 the residual stream *is* the un-normalised embedding
    output. Both are therefore returned.
    """
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    x2d = x.reshape(-1, hidden_dim)
    num_tokens = x2d.shape[0]

    workspace, max_token_num = get_fi_ar_workspace(hidden_dim, x.dtype)
    zeros = _ZERO_RESIDUALS.get((hidden_dim, x.dtype))
    # ``zeros is None`` can only happen if this op runs before the workspace was
    # built; falling back avoids ever allocating it inside a capture.
    if workspace is None or zeros is None or num_tokens > max_token_num:
        out = torch.ops.fastkernels.custom_all_reduce(x)
        if weight_bias == 1.0:
            from ..L1.gemma_rms_norm import GemmaRMSNorm

            return GemmaRMSNorm._forward_static_no_residual(weight, eps, out), out

        from ..L1.rms_norm import RMSNorm

        return RMSNorm.forward_cuda(out, weight, eps), out

    x2d = x2d.contiguous()
    norm_out = torch.empty_like(x2d)
    ar_out = torch.empty_like(x2d)
    _flashinfer_comm.allreduce_fusion(
        input=x2d,
        workspace=workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        output=None,
        residual_out=ar_out,
        norm_out=norm_out,
        residual_in=zeros[:num_tokens],
        rms_gamma=weight,
        rms_eps=eps,
        weight_bias=weight_bias,
        use_oneshot=None,
        fp32_acc=True,
        # See fused_allreduce_add_rmsnorm: with AUTO one-shot selection the only
        # safe value is True.
        trigger_completion_at_end=True,
    )
    return norm_out.view(orig_shape), ar_out.view(orig_shape)
