"""EfficientNetV2 squeeze-and-excitation block."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.global_avg_pool2d import GlobalAvgPool2d
from ..L1.sigmoid import Sigmoid
from ..L1.silu import SiLU


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()
        self.global_pool = GlobalAvgPool2d(keepdim=True)
        self.conv_reduce = Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        self.act1 = SiLU()
        self.conv_expand = Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)
        self.gate = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.global_pool(x)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.gate(scale)
        return x * scale
