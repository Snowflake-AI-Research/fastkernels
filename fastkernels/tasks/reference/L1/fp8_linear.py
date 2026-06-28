"""Semantic PyTorch reference for fp8_linear.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.

Limitations: DeepGEMM, FlashInfer, and custom quantization kernels are replaced
with explicit PyTorch dequantize/matmul/quantize steps. This preserves the
mathematical contract, not CUDA graph behavior or backend-specific layouts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_GROUP_SIZE = 128


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _expand_weight_scale(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight_fp8.shape[-2], weight_fp8.shape[-1]
    row_blocks = _ceil_div(rows, _GROUP_SIZE)
    col_blocks = _ceil_div(cols, _GROUP_SIZE)
    scale_f = scale.float()
    if scale_f.shape[-2:] == (row_blocks, col_blocks):
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-2)
        expanded = expanded.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :rows, :cols]
    if scale_f.shape[-1] == col_blocks:
        expanded = scale_f.repeat_interleave(_GROUP_SIZE, dim=-1)
        return expanded[..., :cols].unsqueeze(-2).expand_as(weight_fp8.float())
    return scale_f.expand_as(weight_fp8.float())


def _quantize_fp8_per_token_group(
    source: torch.Tensor,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
    *,
    use_ue8m0: bool = True,
    eps: float = 1e-10,
) -> None:
    info = torch.finfo(torch.float8_e4m3fn)
    flat = source.reshape(-1, source.shape[-1]).float()
    groups = _ceil_div(flat.shape[-1], _GROUP_SIZE)
    padded_cols = groups * _GROUP_SIZE
    if padded_cols != flat.shape[-1]:
        padded = flat.new_zeros(flat.shape[0], padded_cols)
        padded[:, :flat.shape[-1]] = flat
    else:
        padded = flat
    grouped = padded.view(flat.shape[0], groups, _GROUP_SIZE)
    scale = grouped.abs().amax(dim=-1).clamp_min(eps) / info.max
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    expanded = scale.repeat_interleave(_GROUP_SIZE, dim=-1)[:, :flat.shape[-1]]
    out_fp8.copy_(torch.clamp(flat / expanded, info.min, info.max).to(out_fp8.dtype).view_as(out_fp8))
    out_scale.copy_(scale.view_as(out_scale))


class _Fp8PrefillBufs:
    def __init__(self):
        self.input_fp8 = None
        self.input_scale = None
        self.output = None


class PerTokenGroupQuantFp8(nn.Module):
    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        _quantize_fp8_per_token_group(x, out_fp8, out_scale)


class Fp8Linear(nn.Module):
    BLOCK_SIZE = _GROUP_SIZE
    _FLASHINFER_M_THRESHOLD = 32

    def __init__(self):
        super().__init__()
        self._a_buf = None
        self._s_buf = None
        self._o_buf = None
        self._pf = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = torch.empty(max_tokens, math.ceil(K / _GROUP_SIZE), dtype=torch.float32, device=device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        n, k = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, k)
        weight = weight_fp8.float() * _expand_weight_scale(weight_fp8, weight_scale_inv)
        output = F.linear(input_2d.float(), weight.float(), bias.float() if bias is not None else None)
        return output.to(input_bf16.dtype).view(*input_bf16.shape[:-1], n)


def postprocess_fp8_weights(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return weight_fp8, scale_inv
