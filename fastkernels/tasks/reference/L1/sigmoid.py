"""Semantic PyTorch reference for sigmoid.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
