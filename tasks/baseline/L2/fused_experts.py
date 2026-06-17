"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

Supports both BF16 and FP8 W8A8 block-scaled expert weights.
When DeepGEMM is available (Hopper+ GPUs), uses m_grouped_fp8_gemm_nt_contiguous
with fused SiLU+mul+FP8 quantization between GEMMs. Falls back to the Triton
fused_moe_kernel otherwise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import (
    MoeGroupedGemm,
    _valid_deep_gemm,
    get_triton_config,
    m_grouped_fp8_gemm_nt_contiguous,
)
from ..L1.moe_sum import MoeSum
from ..L1.gelu_and_mul import GeluAndMul
from ..L1.silu_and_mul import SiluAndMul
from ..L1.silu_mul_quant_fp8 import SiluMulQuantFp8

SPARSITY_FACTOR = 4
_FP8_GROUP_SIZE = 128


def _compute_aligned_M(M: int, num_topk: int, local_num_experts: int,
                        alignment: int) -> int:
    """Compute aligned total rows for DeepGEMM."""
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    remainder = M_sum % alignment
    if remainder != 0:
        M_sum += alignment - remainder
    return M_sum


def _deepgemm_permute(
    hidden_states: torch.Tensor,
    a_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute tokens by expert assignment for DeepGEMM contiguous layout.

    Uses vectorized PyTorch ops (scatter_add, argsort) to avoid Python loops.

    Returns:
        (a_perm, a_scale_perm, expert_ids, inv_perm)
    """
    M, K = hidden_states.size()
    top_k = topk_ids.size(1)
    device = hidden_states.device

    M_sum = _compute_aligned_M(M, top_k, local_num_experts, alignment)
    scale_cols = K // _FP8_GROUP_SIZE

    flat_ids = topk_ids.view(-1).to(torch.int64)
    num_tokens_total = flat_ids.size(0)

    expert_num_tokens = torch.zeros(local_num_experts, dtype=torch.int64, device=device)
    expert_num_tokens.scatter_add_(0, flat_ids,
                                   torch.ones(num_tokens_total, dtype=torch.int64, device=device))

    aligned_counts = ((expert_num_tokens + alignment - 1) // alignment) * alignment
    expert_offsets = torch.zeros(local_num_experts + 1, dtype=torch.int64, device=device)
    torch.cumsum(aligned_counts, dim=0, out=expert_offsets[1:])

    # Build expert_ids without host-device sync (.item()) so this is safe
    # inside CUDA graph capture.  For each position in [0, M_sum), determine
    # which expert's aligned block it falls into via searchsorted, then check
    # whether it's within the actual (non-padding) token count.
    pos_idx = torch.arange(M_sum, device=device, dtype=torch.int64)
    # searchsorted(offsets, pos, right=True) - 1 gives the expert whose block
    # contains `pos`.  expert_offsets has E+1 entries (0-based cumsum).
    expert_for_pos = torch.searchsorted(expert_offsets, pos_idx, right=True) - 1
    expert_for_pos = expert_for_pos.clamp_(0, local_num_experts - 1)
    local_pos = pos_idx - expert_offsets[expert_for_pos]
    valid = local_pos < expert_num_tokens[expert_for_pos]
    # Use torch.where (element-wise, fixed output size) instead of boolean
    # indexing which produces data-dependent shapes and breaks CUDA graphs.
    expert_ids = torch.where(valid, expert_for_pos.to(torch.int32),
                             torch.tensor(-1, dtype=torch.int32, device=device))

    sorted_order = torch.argsort(flat_ids, stable=True)

    # Compute within-expert indices using only GPU ops.
    sorted_experts = flat_ids[sorted_order]
    rank_in_sorted = torch.arange(num_tokens_total, device=device, dtype=torch.int64)
    # For each expert, find the first position in sorted order.
    expert_first = torch.full((local_num_experts,), num_tokens_total,
                              dtype=torch.int64, device=device)
    expert_first.scatter_reduce_(0, sorted_experts,
                                 rank_in_sorted, reduce="amin",
                                 include_self=False)
    within_expert_idx = torch.zeros(num_tokens_total, dtype=torch.int64, device=device)
    within_expert_idx[sorted_order] = rank_in_sorted - expert_first[sorted_experts]

    dest_positions = expert_offsets[flat_ids] + within_expert_idx

    a_perm = torch.zeros(M_sum, K, dtype=hidden_states.dtype, device=device)
    a_scale_perm = torch.zeros(M_sum, scale_cols, dtype=torch.float32, device=device)

    token_indices = torch.arange(M, device=device).unsqueeze(1).expand(M, top_k).reshape(-1)

    a_perm[dest_positions] = hidden_states[token_indices]
    a_scale_perm[dest_positions] = a_scale[token_indices]

    inv_perm = dest_positions.view(M, top_k).to(torch.int32)

    return a_perm, a_scale_perm, expert_ids, inv_perm


def _deepgemm_unpermute_and_reduce(
    mm2_out: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Unpermute DeepGEMM output and reduce across top-k experts.

    Uses vectorized gather + weighted sum to avoid Python loops.
    """
    M, K = output.size()
    top_k = topk_ids.size(1)

    flat_positions = inv_perm.to(torch.int64).view(-1)
    gathered = mm2_out[flat_positions].view(M, top_k, K)
    weights = topk_weights.unsqueeze(-1)
    output.copy_((gathered.to(output.dtype) * weights).sum(dim=1))


class _SharedBuf:
    """Mutable container so all FusedExperts layers share one set of scratch
    buffers. Layers execute sequentially so reuse is safe."""
    __slots__ = ("cache13", "a_fp8_1", "a_scale_1", "a_fp8_2", "a_scale_2",
                 "dg_ws1", "dg_ws2")
    def __init__(self):
        self.cache13 = None
        self.a_fp8_1 = None
        self.a_scale_1 = None
        self.a_fp8_2 = None
        self.a_scale_2 = None
        self.dg_ws1 = None
        self.dg_ws2 = None

_SHARED_BUF = _SharedBuf()


class FusedExperts(nn.Module):
    """Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

    On Hopper+ GPUs with DeepGEMM available and valid shapes:
      permute -> DeepGEMM GEMM1 -> fused SiLU+mul+FP8 quant -> DeepGEMM GEMM2 -> unpermute
    Otherwise (Triton fallback):
      MoeAlign -> Triton grouped GEMM1 -> SiLU+mul -> FP8 quant -> Triton grouped GEMM2 -> MoeSum
    """

    def __init__(self, activation: str = "silu", config_style: str = "legacy"):
        super().__init__()
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"Unsupported MoE activation: {activation}")
        if config_style not in ("legacy", "vllm"):
            raise ValueError(f"Unsupported MoE config style: {config_style}")
        self.activation = activation
        self.config_style = config_style
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.act_fn = SiluAndMul() if activation == "silu" else GeluAndMul("tanh")
        self.moe_sum = MoeSum()
        self.per_token_group_quant_fp8 = PerTokenGroupQuantFp8()
        self.silu_mul_quant_fp8 = SiluMulQuantFp8()
        self._sb = _SHARED_BUF

    def _get_cache13(self, total_elems, device, dtype):
        sb = self._sb
        if sb.cache13 is None or sb.cache13.numel() < total_elems:
            sb.cache13 = torch.empty(total_elems, device=device, dtype=dtype)
        return sb.cache13[:total_elems]

    def _get_fp8_bufs(self, buf_id, M, K, device):
        sb = self._sb
        attr_a = f"a_fp8_{buf_id}"
        attr_s = f"a_scale_{buf_id}"
        num_groups = math.ceil(K / _FP8_GROUP_SIZE)
        existing_a = getattr(sb, attr_a)
        if existing_a is None or existing_a.size(0) < M or existing_a.size(1) < K:
            setattr(sb, attr_a, torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device))
            setattr(sb, attr_s, torch.empty(M, num_groups, dtype=torch.float32, device=device))
        a = getattr(sb, attr_a)
        s = getattr(sb, attr_s)
        return a[:M, :K], s[:M, :num_groups]

    def _get_dg_workspace(self, buf_id, shape, device, dtype):
        sb = self._sb
        attr = f"dg_ws{buf_id}"
        existing = getattr(sb, attr)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        needed_bytes = elem_size
        for s in shape:
            needed_bytes *= s
        if existing is None or existing.numel() < needed_bytes:
            setattr(sb, attr, torch.empty(needed_bytes, device=device, dtype=torch.uint8))
        raw = getattr(sb, attr)
        needed_elems = needed_bytes // elem_size
        return raw[:needed_bytes].view(dtype)[:needed_elems].view(shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        w13_scale_dg: torch.Tensor | None = None,
        w2_scale_dg: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ) -> torch.Tensor:
        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        if (self.activation == "silu"
                and use_fp8_w8a8
                and _valid_deep_gemm(hidden_states, w13, w2)
                and not torch.cuda.is_current_stream_capturing()):
            dg_w13_scale = w13_scale_dg if w13_scale_dg is not None else w13_scale
            dg_w2_scale = w2_scale_dg if w2_scale_dg is not None else w2_scale
            return self._forward_deep_gemm(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, dg_w13_scale, dg_w2_scale, block_shape,
                M, K, E, N, N2, top_k,
            )
        else:
            return self._forward_triton(
                hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, w13_scale, w2_scale,
                use_fp8_w8a8, block_shape,
                M, K, E, N, N2, top_k,
            )

    def _forward_deep_gemm(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        """DeepGEMM path: permute -> grouped GEMM1 -> fused act+quant -> grouped GEMM2 -> unpermute."""
        alignment = _FP8_GROUP_SIZE

        M_sum = _compute_aligned_M(M, top_k, num_experts, alignment)

        a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
        self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)

        a1_perm, a1_scale_perm, expert_ids, inv_perm = _deepgemm_permute(
            a_fp8, a_scale, topk_ids, num_experts, alignment,
        )

        mm1_out = self._get_dg_workspace(1, (M_sum, N2), hidden_states.device, hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (a1_perm, a1_scale_perm), (w13, w13_scale), mm1_out, expert_ids,
        )

        quant_out = self._get_dg_workspace(
            2, (M_sum, N), hidden_states.device, torch.float8_e4m3fn,
        )
        a2_fp8, a2_scale = self.silu_mul_quant_fp8(
            mm1_out, output=quant_out,
        )

        mm2_out = self._get_dg_workspace(1, (M_sum, K), hidden_states.device, hidden_states.dtype)
        m_grouped_fp8_gemm_nt_contiguous(
            (a2_fp8, a2_scale), (w2, w2_scale), mm2_out, expert_ids,
        )

        output = torch.empty(M, K, dtype=hidden_states.dtype, device=hidden_states.device)
        _deepgemm_unpermute_and_reduce(mm2_out, topk_ids, topk_weights, inv_perm, output)
        return output

    def _forward_triton(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale,
        use_fp8_w8a8, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        """Triton fallback path (original implementation with JSON autotuning)."""
        config = get_triton_config(
            M, w13.shape, w2.shape, top_k,
            use_fp8=use_fp8_w8a8, block_shape=block_shape,
            default_style=self.config_style,
        )

        use_naive = (M * top_k * SPARSITY_FACTOR <= num_experts)

        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, config["BLOCK_SIZE_M"], num_experts, naive=use_naive,
        )

        cache13_size = M * top_k * max(N2, K)
        cache13_flat = self._get_cache13(cache13_size, hidden_states.device, hidden_states.dtype)
        intermediate1 = cache13_flat[:M * top_k * N2].view(M * top_k, N2)
        intermediate3 = cache13_flat[:M * top_k * K].view(M * top_k, K)

        if use_fp8_w8a8:
            a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
            self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)
            gemm1_input = a_fp8
            gemm1_a_scale = a_scale
        else:
            gemm1_input = hidden_states
            gemm1_a_scale = None

        self.moe_grouped_gemm(
            gemm1_input, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
            a_scale=gemm1_a_scale, b_scale=w13_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        intermediate2 = self.act_fn(intermediate1)

        if use_fp8_w8a8:
            a2_fp8, a2_scale = self._get_fp8_bufs(2, M * top_k, N, hidden_states.device)
            self.per_token_group_quant_fp8(intermediate2, a2_fp8, a2_scale)
            gemm2_input = a2_fp8
            gemm2_a_scale = a2_scale
        else:
            gemm2_input = intermediate2
            gemm2_a_scale = None

        self.moe_grouped_gemm(
            gemm2_input, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
            a_scale=gemm2_a_scale, b_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        return self.moe_sum(intermediate3, top_k)
