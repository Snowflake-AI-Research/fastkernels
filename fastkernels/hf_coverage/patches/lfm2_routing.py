"""GroupedTopK normalization with HF's low-precision boundaries and epsilon."""

import torch
import triton
import triton.language as tl
from torch import nn

from fastkernels.tasks.baseline.L1.grouped_topk import GroupedTopK, _C
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid


@triton.jit
def _normalize(weights, output, K: tl.constexpr, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    kind = output.dtype.element_ty
    values = tl.load(weights + row * K + col, col < K, 0).to(kind).to(tl.float32)
    total = tl.sum(values, 0).to(kind).to(tl.float32)
    denominator = (total + 1e-6).to(kind).to(tl.float32)
    normalized = (values / denominator).to(kind).to(tl.float32)
    tl.store(output + row * K + col, normalized * SCALE, col < K)


class Lfm2Routing(nn.Module):
    """Retain grouped selection and row reduction; change casts and +1e-6 only.

    The parent's existing no-activation CUDA entry accepts already-rounded
    sigmoid scores. This preserves HF's score rounding before expert selection.
    The real extra normalization launch is included in model execution.
    """

    def __init__(self, topk, scale):
        super().__init__()
        self.topk, self.scale = topk, scale
        self.sigmoid = Sigmoid()
        self.cpu_router = GroupedTopK(scoring_func="sigmoid", renormalize=False)

    def forward(self, logits, bias):
        if not logits.is_cuda:
            weights, indices = self.cpu_router(logits, bias, 1, 1, self.topk)
            weights = weights.to(logits.dtype)
            return weights / (weights.sum(-1, keepdim=True) + 1e-6) * self.scale, indices
        weights, indices = _C.grouped_topk(self.sigmoid(logits), 1, 1, self.topk, False, 1.0, bias, 0)
        output = torch.empty_like(weights, dtype=logits.dtype)
        _normalize[(weights.shape[0],)](weights, output, self.topk, triton.next_power_of_2(self.topk), self.scale)
        return output, indices
