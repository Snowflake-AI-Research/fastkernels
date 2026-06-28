"""Universal Inverted Residual (UIB) block for MobileNetV4.

The UIB is the primary building block of MobileNetV4, supporting multiple
configurations (ExtraDW, ConvNeXt, FFN) via optional depthwise branches:

    dw_start -> pw_exp -> dw_mid -> pw_proj -> dw_end -> (+residual)

Unused branches (kernel_size=0) become Identity. The block name and
submodule naming (dw_start, pw_exp, dw_mid, pw_proj, dw_end) match
timm's UniversalInvertedResidual for direct state dict compatibility.

Reference: timm/models/_efficientnet_blocks.py UniversalInvertedResidual
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.relu import ReLU


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round a value to the nearest multiple of divisor."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormAct(nn.Module):
    """Conv2d + BatchNorm2d + optional activation.

    Submodule naming (.conv, .bn) matches timm's ConvNormAct for
    weight-compatible state dicts.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        apply_act: bool = True,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_chs, out_chs, kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = BatchNorm2d(out_chs)
        self.act = ReLU() if apply_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UniversalInvertedResidual(nn.Module):
    """Universal Inverted Residual block (UIB) for MobileNetV4.

    Args:
        in_chs: Input channels.
        out_chs: Output channels.
        dw_kernel_size_start: Kernel size for start depthwise (0 = skip).
        dw_kernel_size_mid: Kernel size for middle depthwise (0 = skip).
        dw_kernel_size_end: Kernel size for end depthwise (0 = skip).
        stride: Stride for the strided depthwise conv.
        exp_ratio: Expansion ratio for pointwise expansion.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size_start: int = 0,
        dw_kernel_size_mid: int = 3,
        dw_kernel_size_end: int = 0,
        stride: int = 1,
        exp_ratio: float = 1.0,
    ):
        super().__init__()
        self.has_skip = (in_chs == out_chs and stride == 1)
        mid_chs = _make_divisible(in_chs * exp_ratio)

        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            self.dw_start = ConvNormAct(
                in_chs, in_chs, dw_kernel_size_start,
                stride=dw_start_stride, groups=in_chs, apply_act=False,
            )
        else:
            self.dw_start = nn.Identity()

        self.pw_exp = ConvNormAct(in_chs, mid_chs, 1)

        if dw_kernel_size_mid:
            self.dw_mid = ConvNormAct(
                mid_chs, mid_chs, dw_kernel_size_mid,
                stride=stride, groups=mid_chs,
            )
        else:
            self.dw_mid = nn.Identity()

        self.se = nn.Identity()

        self.pw_proj = ConvNormAct(mid_chs, out_chs, 1, apply_act=False)

        if dw_kernel_size_end:
            dw_end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            self.dw_end = ConvNormAct(
                out_chs, out_chs, dw_kernel_size_end,
                stride=dw_end_stride, groups=out_chs, apply_act=False,
            )
        else:
            self.dw_end = nn.Identity()

        self.layer_scale = nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
