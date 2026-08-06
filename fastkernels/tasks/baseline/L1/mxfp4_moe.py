"""MXFP4-native fused MoE primitive backed by the OAI Triton kernels.

This module is the L1 wrapper around ``triton_kernels.matmul_ogs`` for
MXFP4-quantized expert weights with OAI-style SwiGLU activation. It owns
all of the routing/quantization/swizzling logic that GPT-OSS needs so
that the L2 ``GptOssMoE`` module can stay pure-composition.

Why we copy this code: the implementations of weight swizzling, routing
data construction, and the fused matmul wrapper live inside vLLM. FastKernels
L2+ modules are not allowed to call into vLLM, so the relevant bits of
``vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe`` and
``vllm.model_executor.layers.quantization.utils.mxfp4_utils`` are
re-implemented here verbatim (modulo cleanup of code paths FastKernels does
not exercise -- AITER/ROCm fallbacks, expert parallelism, w4a8, and the
``use_legacy_triton_kernels`` shim).

The underlying ``triton_kernels`` package is OpenAI's standalone Triton
helper library (https://github.com/triton-lang/triton/tree/main/python/triton_kernels);
it is bundled inside vLLM's ``third_party`` directory but is otherwise
an external dependency. We locate it via the vLLM install path purely
to extend ``sys.path`` -- we never invoke any vLLM function.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# triton_kernels availability
# ---------------------------------------------------------------------------


@functools.cache
def _ensure_triton_kernels_on_path() -> None:
    """Ensure the ``triton_kernels`` package is importable via vLLM's bundle.

    Mirrors vLLM's ``import_triton_kernels`` shim but performs only
    filesystem / sys.path manipulation -- no vLLM functions are called.
    """
    vllm_spec = importlib.util.find_spec("vllm")
    third_party = os.path.join(os.path.dirname(vllm_spec.origin), "third_party")
    if third_party not in sys.path:
        sys.path.insert(0, third_party)


# ---------------------------------------------------------------------------
# Quant config (replacement for vLLM's FusedMoEQuantConfig)
# ---------------------------------------------------------------------------


@dataclass
class Mxfp4MoEQuantConfig:
    """Minimal quant config carrying the per-MoE precision/bias tensors.

    Attribute names match the subset of ``FusedMoEQuantConfig`` consumed
    by ``triton_kernel_fused_experts`` (``w{1,2}_precision`` and
    ``w{1,2}_bias``), so the call sites stay essentially unchanged.
    """

    w1_precision: Any  # triton_kernels.matmul_ogs.PrecisionConfig
    w2_precision: Any  # triton_kernels.matmul_ogs.PrecisionConfig
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Weight swizzling
# ---------------------------------------------------------------------------


def _swizzle_mxfp4(quant_tensor: torch.Tensor, scale: torch.Tensor, num_warps: int):
    """Swizzle MXFP4 weight + E8M0 scales into the layout matmul_ogs wants.

    Returns ``(packed_tensor, in_flex_data, scale_tensor)`` where the two
    tensor returns are ``triton_kernels.tensor.Tensor`` wrappers, ready
    to be plugged into a ``PrecisionConfig``.

    Copied from ``vllm.model_executor.layers.quantization.utils.mxfp4_utils._swizzle_mxfp4``,
    minus the ROCm/Hopper-old-torch fallbacks that FastKernels does not exercise.
    """
    _ensure_triton_kernels_on_path()
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout

    cap = torch.cuda.get_device_capability()

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )

    if cap[0] == 9:
        opt_flags.update_opt_flags_constraints({"split_k": 1})
    elif cap[0] == 10:
        opt_flags.update_opt_flags_constraints(
            {"is_persistent": True, "epilogue_subtile": 1}
        )

    # transpose so the quantization axis is on dim 1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4),
        value_layout,
        **value_layout_opts,
    )
    scale = convert_layout(
        wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
    )
    return quant_tensor, InFlexData(), scale


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _routing_from_logits(logits: torch.Tensor, n_expts_act: int, sm_first: bool):
    """Compute ``(RoutingData, GatherIndx, ScatterIndx)`` from gating logits.

    Delegates to ``triton_kernels.routing.routing``, which fuses softmax, top-k,
    bitmatrix packing and routing-metadata construction into a single launch.
    This is the same entry point vLLM's GPT-OSS MoE uses; earlier revisions of
    this file reimplemented it against a ``SparseMatrix`` /
    ``make_ragged_tensor_metadata`` API that has since been removed from
    ``triton_kernels``.
    """
    _ensure_triton_kernels_on_path()
    from triton_kernels.routing import routing

    return routing(logits, n_expts_act, sm_first=sm_first)


# ---------------------------------------------------------------------------
# Fused experts
# ---------------------------------------------------------------------------


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """Shrink ``x`` and reshape it to ``v``. Used for intermediate caches."""
    n = 1
    for d in v:
        n *= d
    assert n <= x.numel(), f"{v} ({n}) <= {x.shape} ({x.numel()})"
    return x.flatten()[:n].view(*v)


# ``matmul_ogs`` reads its ragged operands a full BLOCK_M tile at a time, so a
# buffer sized to exactly ``M`` (or ``M * topk``) rows is read past its end.
# vLLM never trips this because its MoE buffers come from a workspace sized for
# ``max_num_batched_tokens`` and are then narrowed with ``_resize_cache``; the
# over-read stays inside the workspace.  Allocating exactly, as this module did,
# leaves the tail inside whatever the caching allocator happened to place next:
# harmless with slack, an ``illegal memory access`` once the small allocation
# lands at the end of a segment.  gpt-oss-120b hit that on its first
# single-token decode (M=1, 4 gates -> an 11 KiB intermediate cache), while the
# 16-token prefill in the same process was large enough to absorb it.
#
# 128 is the largest ``block_m`` ``matmul_ogs_details.opt_flags`` picks on
# NVIDIA (``block_m = max(16, min(next_power_of_2(tokens_per_expt), 128))``), so
# rounding the row count up to it keeps every tile inside our own allocation.
_MATMUL_OGS_ROW_TILE = 128


def _tile_rows(rows: int) -> int:
    """Round a ragged-operand row count up to ``matmul_ogs``'s tile."""
    tile = _MATMUL_OGS_ROW_TILE
    return ((rows + tile - 1) // tile) * tile


def _fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    routing_data,
    gather_indx,
    scatter_indx,
    topk: int,
    quant_config: Mxfp4MoEQuantConfig,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Run the two fused MXFP4 matmuls with OAI SwiGLU in between."""
    _ensure_triton_kernels_on_path()
    import triton_kernels.swiglu
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs

    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32
    assert hidden_states.ndim == 2
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    batch_dim = 1
    M, K = hidden_states.shape[-2:]
    _, _, N = w1.shape

    intermediate_cache = torch.empty(
        (batch_dim, _tile_rows(M * topk), N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache = _resize_cache(intermediate_cache, (batch_dim, M * topk, N // 2))
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    # ``reduction_n`` is an argument of ``FusedActivation`` (positional, after
    # the activation args), not of ``FnSpecs``, in the bundled triton_kernels.
    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
    )
    gammas = routing_data.gate_scal if routing_data else None

    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate_cache,
    )
    matmul_ogs(
        intermediate_cache.view(M * topk, N // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    return output_tensor.view(M, K)


# ---------------------------------------------------------------------------
# Public nn.Module interface
# ---------------------------------------------------------------------------


class Mxfp4MoE(nn.Module):
    """MXFP4-quantized fused MoE primitive (routing + matmul_ogs experts).

    The module is stateless -- expert weights, biases, and the
    :class:`Mxfp4MoEQuantConfig` are passed to ``forward`` so a single
    instance can serve any number of MoE layers. Weight preparation is
    exposed as static helpers so the L2 caller does not need to import
    ``triton_kernels`` directly.
    """

    @staticmethod
    def prepare_weight(
        quant_tensor: torch.Tensor,
        scale: torch.Tensor,
        num_warps: int = 8,
    ):
        """Swizzle an MXFP4 expert weight and build its ``PrecisionConfig``.

        Returns ``(swizzled_weight, precision_config)`` ready to feed
        into :meth:`make_quant_config` and :meth:`forward`.
        """
        _ensure_triton_kernels_on_path()
        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

        weight, flex, scale_tensor = _swizzle_mxfp4(quant_tensor, scale, num_warps)
        precision = PrecisionConfig(
            weight_scale=scale_tensor, flex_ctx=FlexCtx(rhs_data=flex)
        )
        return weight, precision

    @staticmethod
    def make_quant_config(
        w1_precision: Any,
        w2_precision: Any,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> Mxfp4MoEQuantConfig:
        """Construct an MXFP4 W4A16 quant config from per-expert precisions/biases."""
        return Mxfp4MoEQuantConfig(
            w1_precision=w1_precision,
            w2_precision=w2_precision,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1,
        w2,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        quant_config: Mxfp4MoEQuantConfig,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """End-to-end MXFP4 MoE forward (routing + fused experts).

        ``w1``/``w2`` must already be swizzled (see :meth:`prepare_weight`)
        and ``quant_config`` must carry the matching precision configs and
        expert biases. ``hidden_states`` must be bfloat16 and 2D.
        """
        routing_data, gather_idx, scatter_idx = _routing_from_logits(
            gating_output, topk, sm_first=not renormalize
        )
        # Over-allocate the output rows for the same reason as the intermediate
        # cache above; ``_fused_experts`` narrows it back to ``M`` rows.
        output = torch.empty(
            (_tile_rows(hidden_states.shape[0]), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return _fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data,
            gather_idx,
            scatter_idx,
            topk=topk,
            quant_config=quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
