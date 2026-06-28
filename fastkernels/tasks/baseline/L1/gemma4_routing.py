"""Gemma4 MoE routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemma4_routing_kernel(
    logits_ptr,
    scale_ptr,
    weights_ptr,
    ids_ptr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    logits = tl.load(
        logits_ptr + pid * E + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    max_l = tl.max(logits, axis=0)

    # Sort logits descending with deterministic expert-id tie-breaking.
    min32 = -2147483648
    bits = logits.to(tl.int32, bitcast=True)
    sign = bits >> 31
    key = tl.where(sign == 0, bits ^ -1, bits ^ min32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    packed = ((key.to(tl.int64) & 0x00000000FFFFFFFF) << 32) | offs_e.to(tl.int64)
    sorted_packed = tl.sort(packed, descending=False)

    sorted_keys = ((sorted_packed >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    sorted_ids = (sorted_packed & 0x00000000FFFFFFFF).to(tl.int32)
    sorted_sign = sorted_keys >> 31
    sorted_bits = tl.where(sorted_sign < 0, sorted_keys ^ -1, sorted_keys ^ min32)
    sorted_logits = sorted_bits.to(tl.float32, bitcast=True)

    top_mask = offs_e < K
    raw_exp = tl.math.exp2((sorted_logits - max_l) * 1.4426950408889634)
    denom = tl.sum(tl.where(top_mask, raw_exp, 0.0), axis=0)
    denom = tl.where(denom > 0.0, denom, 1.0)

    scales = tl.load(
        scale_ptr + sorted_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)
    weights = (raw_exp / denom * scales).to(tl.float32)

    out = pid * K + offs_e
    tl.store(ids_ptr + out, sorted_ids, mask=top_mask)
    tl.store(weights_ptr + out, weights, mask=top_mask)


def _gemma4_routing_triton(
    router_logits: torch.Tensor,
    top_k: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    router_logits = router_logits.contiguous()
    per_expert_scale = per_expert_scale.contiguous()
    tokens, num_experts = router_logits.shape
    weights = torch.empty(
        tokens, top_k, dtype=torch.float32, device=router_logits.device,
    )
    ids = torch.empty(
        tokens, top_k, dtype=torch.int32, device=router_logits.device,
    )
    _gemma4_routing_kernel[(tokens,)](
        router_logits,
        per_expert_scale,
        weights,
        ids,
        num_experts,
        top_k,
        triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return weights, ids


class Gemma4Routing(nn.Module):
    """Softmax over all experts, top-k select, then top-k renormalize."""

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        per_expert_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if router_logits.is_cuda:
            return _gemma4_routing_triton(
                router_logits, top_k, per_expert_scale,
            )
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(probs, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * per_expert_scale[topk_ids].float()
        return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()
