"""ConvNeXtV2 block."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.gelu import GELU
from ..L1.grn import GRN
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


class ConvNeXtV2Layer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dwconv = Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = LayerNorm(dim, eps=1e-6, promote_fp32=False)
        self.pwconv1 = Linear(dim, hidden_dim)
        self.act = GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = Linear(hidden_dim, dim)
        self.drop_path = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        if self.drop_path is not None:
            x = self.drop_path(x)
        return residual + x
