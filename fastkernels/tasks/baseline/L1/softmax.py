"""Softmax / LogSoftmax activations (via torch.nn.functional)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


class LogSoftmax(nn.Module):
    """Numerically-stable log-softmax. Used by the TTT-E2E inner-loop CE loss."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.dim)
