"""Tanh activation: (e^x - e^-x) / (e^x + e^-x)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
