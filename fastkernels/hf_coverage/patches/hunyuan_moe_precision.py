"""Hunyuan's rounded down projection and FP32 routed-expert reduction.

Parents: L1.moe_grouped_gemm._fused_moe_kernel, L2.FusedExperts and L1.MoeSum.
The GEMM2 adaptation retains the parent's tiled dot reduction, expert/token
assignment, launch configuration and output layout. Only its epilogue changes:
round the down projection to the activation dtype, then multiply FP32 routing
weights and store FP32 contributions. The selected HF grouped_mm expert backend
uses this order. Quantized/FP16 branches are deliberately outside this adaptation.

FusedExperts retains its alignment, GEMM1 and activation operations. Its weighted
GEMM2 buffer becomes FP32. MoeSum selects its existing generic sum_out backend
instead of its specialized TOPK8 reduction, retaining the same axis and reusable
output storage. Cast to activation dtype only after reduction. All allocations,
conversions, launches and reductions are part of forward execution.
"""

import torch
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.moe_grouped_gemm import get_triton_config
from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
from fastkernels.tasks.baseline.L2.fused_experts import FusedExperts, SPARSITY_FACTOR


@triton.jit
def _weighted_down_kernel(
    a_ptr, b_ptr, c_ptr, weights_ptr, sorted_ids_ptr, expert_ids_ptr, padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr, NAIVE_BLOCK_ASSIGNMENT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    if NAIVE_BLOCK_ASSIGNMENT:
        offs_token = tl.where(offs_m == 0, pid_m, num_valid_tokens)
    else:
        offs_token = tl.load(sorted_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # GEMM2 already has one activation row per token/expert pair (parent top_k=1).
    a_ptrs = a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + off_expert * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k * BLOCK_SIZE_K + offs_k) < K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator = tl.dot(a.to(compute_type), b.to(compute_type), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Native BF16 down-projection store precedes FP32 route multiplication.
    rounded = accumulator.to(compute_type).to(tl.float32)
    weights = tl.load(weights_ptr + offs_token, mask=token_mask, other=0.0)
    weighted = rounded * weights[:, None]
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, weighted, token_mask[:, None] & (offs_cn[None, :] < N))


def _weighted_down(activation, weight, output, routing_weights, alignment, config):
    sorted_ids, expert_ids, padded = alignment
    naive = sorted_ids is None
    if naive:
        rows = expert_ids.numel() * config['BLOCK_SIZE_M']
    else:
        rows = sorted_ids.size(0)
        if activation.size(0) < config['BLOCK_SIZE_M']:
            rows = min(rows, activation.size(0) * config['BLOCK_SIZE_M'])
    grid = (triton.cdiv(rows, config['BLOCK_SIZE_M']) *
            triton.cdiv(weight.size(1), config['BLOCK_SIZE_N']),)
    launch = {name: config[name] for name in ('num_warps', 'num_stages') if name in config}
    _weighted_down_kernel[grid](
        activation, weight, output, routing_weights,
        sorted_ids if sorted_ids is not None else activation, expert_ids, padded,
        weight.size(1), weight.size(2), rows, activation.size(0),
        activation.stride(0), activation.stride(1),
        weight.stride(0), weight.stride(2), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=config['BLOCK_SIZE_M'], BLOCK_SIZE_N=config['BLOCK_SIZE_N'],
        BLOCK_SIZE_K=config['BLOCK_SIZE_K'], GROUP_SIZE_M=config['GROUP_SIZE_M'],
        compute_type=tl.bfloat16 if activation.dtype == torch.bfloat16 else tl.float32,
        NAIVE_BLOCK_ASSIGNMENT=naive, **launch,
    )


class FP32MoeSum(MoeSum):
    """Select the parent's generic sum_out path for FP32 weighted contributions."""

    def forward(self, values, topk):
        if values.dtype != torch.float32:
            raise TypeError('Hunyuan weighted contributions must remain FP32')
        rows, width = values.size(0) // topk, values.size(1)
        if self._output is None or self._output.size(0) < rows or self._output.size(1) < width:
            self._output = values.new_empty(rows, width)
        output = self._output[:rows, :width]
        return torch.sum(values.view(rows, topk, width), dim=1, out=output)


class HunyuanFusedExperts(FusedExperts):
    """Unquantized BF16/FP32 experts with unchanged FP32 routing weights."""

    def __init__(self):
        super().__init__()
        self.moe_sum = FP32MoeSum()

    def forward(self, hidden_states, w13, w2, topk_weights, topk_ids, num_experts):
        if hidden_states.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError('Hunyuan precision path supports unquantized BF16/FP32 only')
        if w13.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
            raise TypeError('Hunyuan expert weight and activation dtypes must agree')
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise TypeError('Hunyuan route weights must be contiguous FP32')
        tokens, hidden = hidden_states.shape
        topk, twice_intermediate = topk_ids.shape[1], w13.shape[1]
        config = get_triton_config(tokens, w13.shape, w2.shape, topk,
                                   use_fp8=False, block_shape=None,
                                   default_style=self.config_style)
        alignment = self.moe_align(
            topk_ids, config['BLOCK_SIZE_M'], num_experts,
            naive=tokens * topk * SPARSITY_FACTOR <= num_experts,
        )
        projection = self._get_cache13(
            tokens * topk * twice_intermediate, hidden_states.device, hidden_states.dtype,
        ).view(tokens * topk, twice_intermediate)
        self.moe_grouped_gemm(
            hidden_states, w13, projection, topk_weights, *alignment,
            mul_routed_weight=False, top_k=topk, config=config,
        )
        activated = self.act_fn(projection)
        weighted = torch.empty(tokens * topk, hidden, device=hidden_states.device,
                               dtype=torch.float32)
        _weighted_down(activated, w2, weighted, topk_weights, alignment, config)
        return self.moe_sum(weighted, topk).to(hidden_states.dtype)
