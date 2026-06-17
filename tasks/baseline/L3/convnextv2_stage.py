"""ConvNeXtV2 stage with optional downsampling."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.layer_norm2d import LayerNorm2d
from ..L2.convnextv2_block import ConvNeXtV2Layer


class ConvNeXtV2Stage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        depth: int,
        hidden_dim: int,
        downsample: bool,
    ):
        super().__init__()
        if downsample:
            self.downsampling_layer = nn.ModuleList([
                LayerNorm2d(in_dim, eps=1e-6),
                Conv2d(in_dim, dim, kernel_size=2, stride=2),
            ])
        else:
            self.downsampling_layer = nn.ModuleList()
        self.layers = nn.ModuleList([ConvNeXtV2Layer(dim, hidden_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.downsampling_layer) > 0:
            x = self.downsampling_layer[0](x)
            x = self.downsampling_layer[1](x)
        for layer in self.layers:
            x = layer(x)
        return x
