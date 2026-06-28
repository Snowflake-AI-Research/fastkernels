"""Vision patch embedding for Qwen VL models.

Flattens 3D video/image patches via Conv3d weight reshaped into a linear projection.

Unified across Qwen2-VL and Qwen3-VL:
  - bias: Qwen2-VL uses bias=False, Qwen3-VL uses bias=True.
"""


from __future__ import annotations


# Inlined from tasks/reference/L1/conv3d.py
import torch.nn as nn


class Conv3d(nn.Module):
    """Conv3D wrapper matching vllm's Conv3dLayer interface."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple[int, ...], stride: tuple[int, ...] | None = None,
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride or kernel_size, bias=bias)
    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        return self.conv(x)


# Inlined from tasks/reference/L1/linear.py
import torch
import torch.nn.functional as F


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = in_channels * temporal_patch_size * patch_size * patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3d(in_channels, embed_dim, kernel, bias=bias)
        self.linear = Matmul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.input_size)
        return self.linear(
            x,
            self.proj.weight.view(self.embed_dim, self.input_size),
            self.proj.bias,
        )
