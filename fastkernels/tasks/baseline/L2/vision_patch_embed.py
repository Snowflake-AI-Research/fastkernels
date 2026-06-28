"""Vision patch embedding for Qwen VL models.

Flattens 3D video/image patches via Conv3d weight reshaped into a linear projection.

Unified across Qwen2-VL and Qwen3-VL:
  - bias: Qwen2-VL uses bias=False, Qwen3-VL uses bias=True.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv3d import Conv3d
from ..L1.linear import Matmul


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
