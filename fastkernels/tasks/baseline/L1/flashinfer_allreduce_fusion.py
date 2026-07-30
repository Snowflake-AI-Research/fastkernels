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


def fi_ar_fusion_available() -> bool:
    """True when FlashInfer exposes ``allreduce_fusion`` and it is not disabled."""
    global _FI_AVAILABLE, _flashinfer_comm, _TorchDistBackend
    if _FI_AVAILABLE is not None:
        return _FI_AVAILABLE
    if os.environ.get("FASTKERNELS_FI_ALLREDUCE_FUSION", "1") == "0":
        _FI_AVAILABLE = False
        return False
    try:
        import flashinfer.comm as flashinfer_comm
        from flashinfer.comm.mnnvl import TorchDistBackend

        _FI_AVAILABLE = hasattr(flashinfer_comm, "allreduce_fusion")
        _flashinfer_comm = flashinfer_comm
        _TorchDistBackend = TorchDistBackend
    except ImportError:
        _FI_AVAILABLE = False
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
            logger.warning(
                "FlashInfer allreduce fusion disabled: no usable workspace (%s)",
                exc2,
            )
            _WORKSPACES[key] = None
            return None, 0

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
             eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce then fused_add_rms_norm -- the path this op replaces.

    Used above the workspace's token bound and whenever the fusion is
    unavailable. Goes through the same two kernels the unfused graph would, so
    falling back costs nothing beyond not being fused.
    """
    from .rms_norm import RMSNorm

    out = torch.ops.fastkernels.custom_all_reduce(x)
    res = residual.clone()
    return RMSNorm.forward_cuda(out, weight, eps, res)


def fused_allreduce_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``all_reduce(x) + residual``, RMSNormed, in one kernel.

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
        return _unfused(x, residual, weight, eps)

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
        # mnnvl sizes its own one-shot workspace around FlashInfer's AUTO
        # strategy, so forcing a choice can request one-shot for a tensor that
        # does not fit it. Let FlashInfer decide (vLLM does the same).
        use_oneshot=None,
        fp32_acc=True,
        # The one-shot Lamport all-reduce signals PDL completion before its
        # output buffer is committed, so a PDL-launched successor can read an
        # uninitialized buffer and produce NaN. Only reachable at small token
        # counts, where one-shot is always chosen -- so complete at the end
        # there. Mirrors vLLM's condition exactly.
        trigger_completion_at_end=num_tokens > _PDL_ADVANCE_LAUNCH_TOKENS,
    )
    return norm_out.view(orig_shape), residual_out.view(residual.shape)
