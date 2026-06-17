"""Squared ReLU activation: max(0, x)^2.

Used by RWKV7's feed-forward block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()
