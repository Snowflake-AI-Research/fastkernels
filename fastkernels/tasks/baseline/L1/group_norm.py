"""GroupNorm wrapping F.group_norm with learnable affine parameters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    """Group normalization with optional learnable weight and bias."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
