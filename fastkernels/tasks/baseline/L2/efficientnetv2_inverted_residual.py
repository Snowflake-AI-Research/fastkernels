"""EfficientNetV2 inverted residual block."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU
from .batch_norm_act2d import BatchNormAct2d
from .efficientnetv2_squeeze_excite import SqueezeExcite


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs: int,
        exp_chs: int,
        out_chs: int,
        stride: int,
        se_reduce_chs: int,
        has_skip: bool,
    ):
        super().__init__()
        self.has_skip = has_skip
        self.conv_pw = Conv2d(in_chs, exp_chs, kernel_size=1, bias=False)
        self.bn1 = BatchNormAct2d(exp_chs, act_layer=SiLU())
        self.conv_dw = Conv2d(exp_chs, exp_chs, kernel_size=3, stride=stride, padding=1, groups=exp_chs, bias=False)
        self.bn2 = BatchNormAct2d(exp_chs, act_layer=SiLU())
        self.aa = None
        self.se = SqueezeExcite(exp_chs, se_reduce_chs)
        self.conv_pwl = Conv2d(exp_chs, out_chs, kernel_size=1, bias=False)
        self.bn3 = BatchNormAct2d(out_chs)
        self.drop_path = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        if self.aa is not None:
            x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = residual + x
        return x
