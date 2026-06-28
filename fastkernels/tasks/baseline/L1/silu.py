"""SiLU (Swish) activation: x * sigmoid(x)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)
