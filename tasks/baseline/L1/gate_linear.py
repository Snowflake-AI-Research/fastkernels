"""DeepSeek MoE router gate matmul (BF16 x BF16 -> FP32) with vLLM parity.

Mirrors vLLM's
``vllm/model_executor/layers/fused_moe/router/gate_linear.py:GateLinear``
which has a three-tier dispatch:

1. **DSV3 specialized kernel** — Hopper/Blackwell, ``num_experts in {256, 384}``,
   ``hidden_size == 7168``, batch ``<= 16``. BF16 x BF16 -> FP32 fused kernel
   that internally accumulates in FP32. Routes to
   ``_C.dsv3_router_gemm`` (verbatim port of vLLM's CUDA kernel — see
   ``tasks/baseline/L1/csrc/dsv3_router_gemm_*.cu``).
2. **cuBLAS BF16 -> FP32** — Hopper/Blackwell + BF16 weight + FP32 out. Routes
   to ``_C.router_gemm_bf16_fp32`` (verbatim port of vLLM's cuBLAS wrapper —
   see ``tasks/baseline/L1/csrc/router_gemm_bf16_fp32.cu``).
3. **PyTorch fallback** — vanilla ``F.linear`` at the input dtype, with cast
   back to FP32 at the end.

Matching vLLM exactly here matters for **correctness** of the grouped-topk
router: kb_nano's previous "promote both to FP32 then matmul" path was
strictly more precise but used a different accumulation order than vLLM's
specialized kernels, which flipped near-tie expert / group selections in
the noaux_tc path (see audit notes).
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

from .csrc import _C


@functools.cache
def _is_hopper_or_blackwell() -> bool:
    """Same gate vLLM uses (see ``GateLinear.__init__``):
    ``current_platform.is_device_capability((9, 0))`` (Hopper) or
    ``current_platform.is_device_capability_family(100)`` (Blackwell)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return (cap[0], cap[1]) == (9, 0) or cap[0] == 10


def _dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Allocates the output and dispatches to the DSV3 specialized kernel.

    Mirrors vLLM's ``_custom_ops.dsv3_router_gemm`` Python wrapper exactly:
    the underlying CUDA op takes ``output`` as an in/out parameter, so the
    allocation lives on the Python side.
    """
    output = torch.empty(
        hidden_states.shape[0],
        router_weight.shape[0],
        device=hidden_states.device,
        dtype=output_dtype,
    )
    _C.dsv3_router_gemm(output, hidden_states, router_weight)
    return output


class GateLinear(nn.Module):
    """DeepSeek MoE router gate matmul with vLLM-parity three-tier dispatch.

    Mirrors the SOTA name (``vllm.../gate_linear.py:GateLinear``).  Stateless;
    the router weight is owned by the parent module and passed through
    ``forward``.
    """

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute router logits with vLLM-parity dispatch.

        Args:
            x: ``(num_tokens, hidden_size)`` activations (BF16).
            weight: ``(num_experts, hidden_size)`` gate weight (BF16).
            out_dtype: Desired output dtype (FP32 for DeepSeek-V3 monolithic).

        Returns:
            ``(num_tokens, num_experts)`` router logits in ``out_dtype``.
        """
        num_tokens = x.shape[0]
        num_experts = weight.shape[0]
        hidden_size = weight.shape[1]

        is_hopper_or_blackwell = _is_hopper_or_blackwell()
        bf16_input = x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

        # Tier 1: DSV3 specialized kernel. Matches vLLM's ``GateLinear.forward``
        # which dispatches here for any ``out_dtype`` (BF16 or FP32) as long as
        # ``batch<=16`` and the hidden/num_experts dims match. Previously we
        # also required ``out_dtype == torch.float32``, which diverted the BF16
        # router path (used by non-monolithic FP8 MoE in DeepSeek-V3.2) to the
        # fallback F.linear — producing slightly different router_logits and
        # boundary-flipping the grouped-topk at layers 3+.
        if (
            is_hopper_or_blackwell
            and bf16_input
            and num_tokens <= 16
            and num_experts in (256, 384)
            and hidden_size == 7168
        ):
            return _dsv3_router_gemm(x, weight, out_dtype)

        # Tier 2: cuBLAS BF16 x BF16 -> FP32.
        if (
            is_hopper_or_blackwell
            and bf16_input
            and out_dtype == torch.float32
        ):
            return _C.router_gemm_bf16_fp32(x, weight)

        # Tier 3: F.linear fallback. Match vLLM's behaviour (cast input to
        # weight dtype, then cast output to out_dtype).
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        out = torch.nn.functional.linear(x, weight)
        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out
