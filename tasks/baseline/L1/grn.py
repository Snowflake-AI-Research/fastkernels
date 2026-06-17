"""Global Response Normalization for ConvNeXtV2-style channels-last activations."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * nx) + self.bias + x
