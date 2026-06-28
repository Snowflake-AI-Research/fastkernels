"""Global average pooling over spatial NCHW dimensions."""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalAvgPool2d(nn.Module):
    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-2, -1), keepdim=self.keepdim)
