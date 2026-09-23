"""Gemma3n statistics from L1 rms_norm_gated's row LayerNorm reduction.

Parent: layer_norm_fwd_kernel, its independent [ROWS, BLOCK_N] tiles, FP32 loads,
row sum / centered-square reduction, and O(M*N) storage and work. This variant
returns row mean/population std instead of affine-normalized vectors. RMS mode
uses the parent's IS_RMS_NORM reduction with explicit input-dtype square/mean
stores and square root, as required by Gemma3n's AltUp magnitude calculation.
No cross-row communication, prefix scan, or new reduction axis is introduced.
CPU is a native diagnostic. RowStats, RowSum, and PrefixSum passed the focused
NVIDIA B200 component checks; full-model acceptance is evaluated separately.
"""

import torch
from torch import nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.rms_norm_gated import calc_rows_per_block


@triton.jit
def _row_stats(X, Mean, Scale, M: tl.constexpr, N: tl.constexpr,
               RMS: tl.constexpr, FLOOR: tl.constexpr,
               BLOCK_N: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    dtype = X.dtype.element_ty
    x = tl.load(X + rows[:, None] * N + cols[None, :], mask, 0.).to(tl.float32)
    if RMS:
        square = (x * x).to(dtype).to(tl.float32)
        variance = (tl.sum(square, 1) / N).to(dtype).to(tl.float32)
        floor = tl.full((), FLOOR, tl.float32).to(dtype).to(tl.float32)
        scale = tl.sqrt(tl.maximum(variance, floor))
        mean = tl.full((ROWS,), 0., tl.float32)
    else:
        mean = tl.sum(x, 1) / N
        centered = tl.where(mask, x - mean[:, None], 0.)
        variance = tl.sum(centered * centered, 1) / N
        scale = tl.sqrt(variance)
    tl.store(Mean + rows, mean, rows < M)
    tl.store(Scale + rows, scale, rows < M)


class Gemma3nRowStats(nn.Module):
    def __init__(self, rms=False, floor=0.):
        super().__init__()
        self.rms, self.floor = rms, floor

    def forward(self, x):
        if not x.is_cuda:
            if self.rms:
                # Native torch.maximum promotes its scalar tensor to x.dtype.
                mean_square = x.square().mean(-1, keepdim=True)
                return torch.zeros_like(mean_square), mean_square.clamp_min(self.floor).sqrt()
            return x.mean(-1, keepdim=True), x.std(-1, keepdim=True, unbiased=False)
        x = x.contiguous()
        n, m = x.shape[-1], x.numel() // x.shape[-1]
        block = triton.next_power_of_2(n)
        if block * x.element_size() > 65536:
            raise ValueError('Gemma3n stats retain the parent 64KB row limit')
        mean, scale = (x.new_empty(x.shape[:-1] + (1,)) for _ in range(2))
        rows = calc_rows_per_block(m, x.device)
        _row_stats[(triton.cdiv(m, rows),)](
            x, mean, scale, m, n, self.rms, self.floor,
            BLOCK_N=block, ROWS=rows, num_warps=min(max(block // 256, 1), 8))
        return mean, scale


@triton.jit
def _row_sum(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + rows[:, None] * N + cols[None, :], mask, 0.).to(tl.float32)
    tl.store(Y + rows, tl.sum(x, 1), rows < M)


class Gemma3nRowSum(nn.Module):
    """The same parent's mean numerator, returned before division by width."""
    def forward(self, x):
        if not x.is_cuda:
            return x.sum(-1, keepdim=True)
        x = x.contiguous()
        n, m = x.shape[-1], x.numel() // x.shape[-1]
        block = triton.next_power_of_2(n)
        if block * x.element_size() > 65536:
            raise ValueError('Gemma3n reduction retains the parent 64KB row limit')
        output = x.new_empty(x.shape[:-1] + (1,))
        rows = calc_rows_per_block(m, x.device)
        _row_sum[(triton.cdiv(m, rows),)](x, output, m, n, BLOCK_N=block, ROWS=rows,
                                       num_warps=min(max(block//256, 1), 8))
        return output


class Gemma3nPrefixSum(nn.Module):
    """Reuse gated_delta_rule.chunk_local_cumsum_scalar unchanged on GPU.

    Timesteps are its existing sequence axis, with one head. Carry each chunk's
    last result into the next using connecting additions; no dense triangular
    matrix or repeated-prefix reduction is introduced. CPU is diagnostic only.
    """
    def forward(self, x):
        from fastkernels.tasks.baseline.L1.gated_delta_rule import chunk_local_cumsum_scalar
        pieces, carry = [], 0.
        for chunk in x.split(256, 1):
            if chunk.is_cuda:
                prefix = chunk_local_cumsum_scalar(chunk.contiguous(), 256)
            else:
                prefix = chunk.cumsum(1)
            prefix = prefix + carry
            pieces.append(prefix)
            carry = prefix[:, -1:]
        return torch.cat(pieces, 1)
