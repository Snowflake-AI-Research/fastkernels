"""DeepSeek-only fused MoE experts mirroring vLLM's ``fused_experts_impl``.

This module exists to give DeepSeek-V3 a path that is bit-identical to
vLLM's reference Triton MoE *and* allocates all scratch buffers fresh
per call (matching vLLM exactly).  The shared-buffer ``FusedExperts`` is
unsuitable here for two reasons:

1.  vLLM's ``select_fp8_moe_backend`` oracle (see
    ``vllm/.../fused_moe/oracle/fp8.py:_get_priority_backends``)
    explicitly forces the ``TRITON`` backend for Hopper + FP8 block-quant
    + TP (no EP).  The DeepGEMM path drifts ~1 BF16 ULP for that
    configuration, which then cascades through near-tie expert
    selection in subsequent MoE layers.  ``FusedExperts`` does honour
    ``prefer_triton=True`` for that, but...
2.  ``FusedExperts`` reuses ``_SHARED_BUF`` and ``MoeAlign`` /
    ``MoeSum`` instance buffers across layers and across decode/prefill
    phases.  CUDA graphs captured during decode warmup hold raw pointers
    into those buffers; if a later eager prefill grows them, the
    captured pointers become dangling and the next decode replay hits
    ``cudaErrorIllegalAddress``.  vLLM avoids this entirely by allocating
    every intermediate (``cache13``, ``intermediate_cache2``,
    ``qhidden_states``, ``a*_scale``, ``sorted_ids``, ``expert_ids``,
    ``num_tokens_post_pad``, ``out_hidden_states``) fresh per call.

The orchestration here is a line-for-line port of
``vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl``
restricted to the ``use_fp8_w8a8 + block_shape`` path that DeepSeek
hits.  All compute kernels are kb_nano L1 primitives that we have
already verified produce bit-identical outputs to vLLM's:

* ``_C.moe_align_block_size``           (kb_nano binding of the same
                                         sgl-kernel CUDA source vLLM uses)
* ``_per_token_group_quant_fp8``        (calls ``torch.ops._C.per_token_group_fp8_quant``)
* ``MoeGroupedGemm._fused_moe_kernel``  (Triton kernel forked from vLLM)
* ``torch.ops._C.silu_and_mul``         (vLLM's packed CUDA kernel; kb_nano
                                         re-uses the same op)
* ``_C.moe_sum``                        (kb_nano CUDA kernel; falls back to
                                         ``at::sum_out`` for top_k > 4 which
                                         matches vLLM's PyTorch reference for
                                         DeepSeek's top_k = 8)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton

import vllm._C  # noqa: F401  - registers torch.ops._C.silu_and_mul + per_token_group_fp8_quant

from ..L1.csrc import _C
from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L1.moe_grouped_gemm import MoeGroupedGemm, get_triton_config

_FP8_GROUP_SIZE = 128
SPARSITY_FACTOR = 4


def _moe_align_block_size_fresh(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh-allocation port of vLLM's ``moe_align_block_size``.

    Mirrors ``vllm.model_executor.layers.fused_moe.moe_align_block_size``
    (no ``expert_map``, ``ignore_invalid_experts=True`` semantics).  All
    output tensors are freshly allocated, matching vLLM and avoiding
    the captured-pointer aliasing trap that bit kb_nano's persistent
    ``MoeAlign`` buffers when used inside CUDA graphs.
    """
    numel = topk_ids.numel()
    if numel < num_experts:
        max_padded = numel * block_size
    else:
        max_padded = numel + num_experts * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)
    device = topk_ids.device

    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    cumsum_buffer = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

    _C.moe_align_block_size(
        topk_ids.view(-1).contiguous(),
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


class VllmFusedExperts(nn.Module):
    """DeepSeek-only Triton MoE that mirrors vLLM's ``fused_experts_impl``.

    Restricted to the ``use_fp8_w8a8 + block_shape`` path (no DeepGEMM,
    no int8/int4, no MX, no expert_map, no router-weight-on-input, SiLU
    activation only).  All scratch buffers are allocated fresh per call
    so the op is safe to mix with CUDA graph capture even when shared
    layer state would otherwise alias.
    """

    def __init__(self):
        super().__init__()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.per_token_group_quant_fp8 = PerTokenGroupQuantFp8()

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        block_shape: list[int],
    ) -> torch.Tensor:
        assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
        assert w13.stride(-1) == 1 and w2.stride(-1) == 1
        assert hidden_states.dtype in (torch.bfloat16, torch.float16, torch.float32)
        assert topk_weights.size() == topk_ids.size()

        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)
        device = hidden_states.device
        dtype = hidden_states.dtype

        config = get_triton_config(
            M, w13.shape, w2.shape, top_k,
            use_fp8=True, block_shape=block_shape,
        )
        block_m = config["BLOCK_SIZE_M"]

        # Mirrors vLLM's ``cache13`` reuse trick: a single flat buffer
        # sized to ``M * top_k * max(N2, K)`` is viewed as
        # ``intermediate_cache1`` (post-GEMM1) and later as
        # ``intermediate_cache3`` (post-GEMM2).  Safe because GEMM2's
        # input (``intermediate_cache2``) is already a separate buffer
        # by the time we write GEMM2's output, so cache1 is dead.
        cache13 = torch.empty(
            M * top_k * max(N2, K), device=device, dtype=dtype,
        )
        intermediate_cache1 = cache13[: M * top_k * N2].view(M * top_k, N2)
        intermediate_cache3 = cache13[: M * top_k * K].view(M * top_k, K)
        intermediate_cache2 = torch.empty(
            (M * top_k, N), device=device, dtype=dtype,
        )

        # Per-token-group FP8 quantize the activations.  Matches
        # vLLM's ``moe_kernel_quantize_input`` -> ``per_token_group_quant_fp8``
        # for ``block_shape=[128, 128]`` (UE8M0 scales on Hopper, same
        # ``torch.ops._C.per_token_group_fp8_quant`` kernel).
        num_groups = math.ceil(K / _FP8_GROUP_SIZE)
        a1_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device)
        a1_scale = torch.empty(M, num_groups, dtype=torch.float32, device=device)
        self.per_token_group_quant_fp8(hidden_states, a1_fp8, a1_scale)

        # Block alignment.  vLLM uses the ``naive_block_assignment`` fast
        # path when ``num_tokens * top_k * SPARSITY_FACTOR <= num_experts``
        # (no expert_map and not int8/int4 block).  For DeepSeek this
        # only triggers at very small decode batch sizes; we still
        # support it because skipping the sort kernel is faster and
        # bit-exact.
        naive = (M * top_k * SPARSITY_FACTOR) <= num_experts
        if naive:
            max_padded = topk_ids.numel() * block_m
            expert_ids = topk_ids.view(-1).to(torch.int32)
            num_tokens_post_padded = torch.empty(
                1, dtype=torch.int32, device=device,
            )
            num_tokens_post_padded.fill_(max_padded)
            sorted_token_ids = None
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                _moe_align_block_size_fresh(topk_ids, block_m, num_experts)
            )

        # GEMM1: a1 @ w13  ->  intermediate_cache1
        # ``mul_routed_weight=False`` matches vLLM's
        # ``apply_router_weight_on_input=False`` default for DeepSeek.
        self.moe_grouped_gemm(
            a1_fp8, w13, intermediate_cache1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
            a_scale=a1_scale, b_scale=w13_scale,
            use_fp8_w8a8=True, block_shape=block_shape,
        )

        # SiLU + mul, in place into ``intermediate_cache2``.  Same
        # packed-bf16x2/half2 kernel vLLM uses
        # (``vllm/csrc/activation_kernels.cu``).
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

        # Quantize the SiLU output for GEMM2.  Same kernel as a1, but
        # the input hidden dim is ``N`` (= ``intermediate_size_per_tp``
        # = 256 for DSV3.2) instead of ``K``.
        num_groups2 = math.ceil(N / _FP8_GROUP_SIZE)
        a2_fp8 = torch.empty(
            M * top_k, N, dtype=torch.float8_e4m3fn, device=device,
        )
        a2_scale = torch.empty(
            M * top_k, num_groups2, dtype=torch.float32, device=device,
        )
        self.per_token_group_quant_fp8(intermediate_cache2, a2_fp8, a2_scale)

        # GEMM2: a2 @ w2  ->  intermediate_cache3.
        # ``mul_routed_weight=True`` and ``top_k=1`` matches vLLM's
        # second ``dispatch_fused_moe_kernel`` call (the routed
        # weight is folded in here, not at GEMM1).
        self.moe_grouped_gemm(
            a2_fp8, w2, intermediate_cache3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
            a_scale=a2_scale, b_scale=w2_scale,
            use_fp8_w8a8=True, block_shape=block_shape,
        )

        # Sum across the top-k dim.  For DeepSeek's top_k=8 this hits
        # the ``default`` branch in ``moe_sum.cu`` which calls
        # ``at::sum_out`` — the same FP-summation order vLLM's
        # ``ops.moe_sum`` uses for top_k > 4.
        out_hidden_states = torch.empty(M, K, device=device, dtype=dtype)
        _C.moe_sum(intermediate_cache3.view(M, top_k, K), out_hidden_states)
        return out_hidden_states
