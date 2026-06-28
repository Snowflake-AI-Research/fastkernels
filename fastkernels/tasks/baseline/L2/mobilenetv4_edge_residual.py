"""EdgeResidual (FusedMBConv) block for MobileNetV4.

Fused expansion convolution followed by pointwise-linear projection.
Used in MobileNetV4 stage 0 (er_* architecture tokens). Submodule
naming (conv_exp, bn1, conv_pwl, bn2) matches timm's EdgeResidual
for direct state dict compatibility.

Reference: timm/models/_efficientnet_blocks.py EdgeResidual
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.relu import ReLU
from .mobilenetv4_uib import _make_divisible


class EdgeResidual(nn.Module):
    """Edge Residual (Fused Inverted Bottleneck) block.

    Args:
        in_chs: Input channels.
        out_chs: Output channels.
        exp_kernel_size: Kernel size for expansion convolution.
        stride: Convolution stride.
        exp_ratio: Expansion ratio.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        exp_kernel_size: int = 3,
        stride: int = 1,
        exp_ratio: float = 1.0,
    ):
        super().__init__()
        mid_chs = _make_divisible(in_chs * exp_ratio)
        self.has_skip = (in_chs == out_chs and stride == 1)

        self.conv_exp = Conv2d(
            in_chs, mid_chs, exp_kernel_size,
            stride=stride,
            padding=exp_kernel_size // 2,
            bias=False,
        )
        self.bn1 = BatchNorm2d(mid_chs)
        self.act1 = ReLU()

        self.conv_pwl = Conv2d(mid_chs, out_chs, 1, bias=False)
        self.bn2 = BatchNorm2d(out_chs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.act1(self.bn1(self.conv_exp(x)))
        x = self.bn2(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x
