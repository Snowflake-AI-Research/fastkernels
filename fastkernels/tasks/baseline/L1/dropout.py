"""Dropout wrapping F.dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=self.training)
