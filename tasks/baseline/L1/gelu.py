"""GELU activation (via torch.nn.functional.gelu).

Supports both exact and approximate ("tanh") modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)
