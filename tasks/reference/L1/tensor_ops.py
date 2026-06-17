"""Primitive tensor manipulation ops.

L1 ops wrapping standard tensor utilities for use by L2+ composites.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pad(nn.Module):
    """Functional padding op."""

    def forward(
        self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0,
    ) -> torch.Tensor:
        return F.pad(x, pad, value=value)


class OneHot(nn.Module):
    """Functional one-hot encoding op."""

    def forward(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(x, num_classes)
