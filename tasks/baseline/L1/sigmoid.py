"""Sigmoid activation: 1 / (1 + exp(-x))."""

from __future__ import annotations

import torch
import torch.nn as nn


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
