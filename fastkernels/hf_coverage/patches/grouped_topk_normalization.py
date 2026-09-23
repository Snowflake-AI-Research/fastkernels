"""Adapt the existing grouped router's selected-weight normalization denominator.

Selection and score computation remain in GroupedTopK. Its existing row-sum
normalization is replaced by the same reduction with an additive epsilon or
denominator floor. Scaling stays before expert execution, as in HF. The extra
launch is part of measured execution; this does not assume hypothetical fusion.
"""

import torch
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.grouped_topk import GroupedTopK


@triton.jit
def _normalize(weights, output, K: tl.constexpr, BLOCK: tl.constexpr,
               EPS: tl.constexpr, FLOOR: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    values = tl.load(weights + row * K + col, col < K, 0).to(tl.float32)
    denominator = tl.maximum(tl.sum(values, 0) + EPS, FLOOR)
    tl.store(output + row * K + col, values / denominator * SCALE, col < K)


class GroupedTopKNormalization(GroupedTopK):
    def __init__(self, *, scoring_func, epsilon=0.0, floor=0.0, scale=1.0):
        super().__init__(scoring_func=scoring_func, renormalize=False)
        self.epsilon, self.floor, self.scale = epsilon, floor, scale

    def forward(self, logits, bias, num_expert_group, topk_group, topk):
        weights, indices = super().forward(logits, bias, num_expert_group, topk_group, topk)
        weights = weights.contiguous()
        output = torch.empty_like(weights)
        _normalize[(weights.shape[0],)](
            weights, output, topk, triton.next_power_of_2(topk),
            self.epsilon, self.floor, self.scale,
        )
        return output, indices
