"""Selected-patch statistics using the existing gated LayerNorm row reduction.

The parent is L1/rms_norm_gated.layer_norm_fwd_kernel. A padding mask weights
the same row sums; this emits mean, population variance, and standard deviation
instead of affine-normalized values. No scan or other-row dependency is added.
"""

import torch
from torch import nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.rms_norm_gated import calc_rows_per_block


@triton.jit
def _masked_stats(X, Padding, Mean, Variance, Std, M: tl.constexpr, N: tl.constexpr,
                  TOL: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    present = (rows[:, None] < M) & (cols[None, :] < N)
    dtype = X.dtype.element_ty
    x = tl.load(X + rows[:, None] * N + cols[None, :], present, 0).to(tl.float32)
    pad = tl.load(Padding + rows[:, None] * N + cols[None, :], present, 1).to(tl.float32)
    mask = 1. - pad
    count = tl.maximum(tl.sum(mask, 1), 1.)
    mean = (tl.sum(x * mask, 1).to(dtype).to(tl.float32) / count).to(dtype).to(tl.float32)
    centered = (x - mean[:, None]).to(dtype).to(tl.float32) * mask
    square = (centered * centered).to(dtype).to(tl.float32)
    variance = tl.maximum((tl.sum(square, 1).to(dtype).to(tl.float32) / count).to(dtype).to(tl.float32), 0.)
    std = tl.maximum(tl.sqrt(variance).to(dtype).to(tl.float32), TOL)
    tl.store(Mean + rows, mean, rows < M)
    tl.store(Variance + rows, tl.maximum(variance, TOL * TOL), rows < M)
    tl.store(Std + rows, std, rows < M)


class MaskedPatchStats(nn.Module):
    def __init__(self, tolerance):
        super().__init__()
        self.tolerance = tolerance

    def forward(self, data, padding):
        if not data.is_cuda:
            mask = 1 - padding
            count = mask.sum(-1).clamp_min(1.)
            mean = (data * mask).sum(-1) / count
            centered = (data - mean[:, None]) * mask
            variance = (centered.square().sum(-1) / count).clamp_min(0.)
            return mean, variance.clamp_min(self.tolerance ** 2), variance.sqrt().clamp_min(self.tolerance)
        data, padding = data.contiguous(), padding.contiguous()
        count, width = data.shape
        mean, variance, std = (data.new_empty(count) for _ in range(3))
        rows = calc_rows_per_block(count, data.device)
        block = triton.next_power_of_2(width)
        if block * data.element_size() > 65536:
            raise ValueError("Masked patch statistics retain the parent's 64KB row limit")
        _masked_stats[(triton.cdiv(count, rows),)](
            data, padding, mean, variance, std, count, width, self.tolerance,
            BLOCK_N=block, ROWS=rows, num_warps=min(max(block // 256, 1), 8))
        return mean, variance, std
