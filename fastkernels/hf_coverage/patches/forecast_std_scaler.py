"""Unmasked forecast scaling adapted from L1 rms_norm_gated's LayerNorm kernel.

Retains its row tiles and mean/variance reductions. The changes preserve HF's
intermediate dtype stores, omit affine parameters, and return mean/std alongside
normalized values. This is only the default all-observed input path.
"""

import torch
from torch import nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.rms_norm_gated import calc_rows_per_block


@triton.jit
def _std_scaler(X, Y, Mean, Std, M: tl.constexpr, N: tl.constexpr,
                EPS: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    dtype = X.dtype.element_ty
    x = tl.load(X + rows[:, None] * N + cols[None, :], mask, 0).to(tl.float32)
    count = tl.full((), N, tl.float32).to(dtype).to(tl.float32)
    total = tl.sum(x, 1).to(dtype).to(tl.float32)
    mean = (total / count).to(dtype).to(tl.float32)
    centered = (x - mean[:, None]).to(dtype).to(tl.float32)
    squares = (centered * centered).to(dtype).to(tl.float32)
    squares = tl.where(mask, squares, 0.)
    total_square = tl.sum(squares, 1).to(dtype).to(tl.float32)
    variance = (total_square / count).to(dtype).to(tl.float32)
    stabilized = (variance + EPS).to(dtype).to(tl.float32)
    std = tl.sqrt(stabilized).to(dtype).to(tl.float32)
    output = centered / std[:, None]
    tl.store(Y + rows[:, None] * N + cols[None, :], output, mask)
    tl.store(Mean + rows, mean, rows < M)
    tl.store(Std + rows, std, rows < M)


class UnmaskedStdScaler(nn.Module):
    def __init__(self, minimum_scale=1e-5):
        super().__init__()
        self.minimum_scale = minimum_scale

    def forward(self, data):
        if not data.is_cuda:
            count = data.new_tensor(data.shape[1])
            mean = data.sum(1, keepdim=True) / count
            centered = data - mean
            variance = centered.square().sum(1, keepdim=True) / count
            std = (variance + self.minimum_scale).sqrt()
            return centered / std, mean, std
        batch, length, channels = data.shape
        x = data.transpose(1, 2).contiguous().reshape(-1, length)
        output = torch.empty_like(x)
        mean, std = (data.new_empty(batch * channels) for _ in range(2))
        rows = calc_rows_per_block(batch * channels, data.device)
        block = triton.next_power_of_2(length)
        if block * data.element_size() > 65536:
            raise ValueError("Forecast scaler retains the parent's 64KB row limit")
        warps = min(max(block // 256, 1), 8)
        _std_scaler[(triton.cdiv(batch * channels, rows),)](
            x, output, mean, std, batch * channels, length, self.minimum_scale,
            BLOCK_N=block, ROWS=rows, num_warps=warps)
        return (output.reshape(batch, channels, length).transpose(1, 2),
                mean.reshape(batch, 1, channels), std.reshape(batch, 1, channels))
