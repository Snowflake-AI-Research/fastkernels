"""EfficientNetV2 EdgeResidual block."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU
from .batch_norm_act2d import BatchNormAct2d


class EdgeResidual(nn.Module):
    def __init__(self, in_chs: int, exp_chs: int, out_chs: int, stride: int, has_skip: bool):
        super().__init__()
        self.has_skip = has_skip
        self.conv_exp = Conv2d(in_chs, exp_chs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNormAct2d(exp_chs, act_layer=SiLU())
        self.aa = None
        self.se = None
        self.conv_pwl = Conv2d(exp_chs, out_chs, kernel_size=1, bias=False)
        self.bn2 = BatchNormAct2d(out_chs)
        self.drop_path = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        if self.aa is not None:
            x = self.aa(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = residual + x
        return x
