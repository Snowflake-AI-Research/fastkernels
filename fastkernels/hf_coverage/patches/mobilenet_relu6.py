"""ReLU with MobileNet's upper clipping boundary at six."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastkernels.tasks.baseline.L1.relu import ReLU


class ReLU6(ReLU):
    """Keep the parent's pointwise activation, adding the finite upper bound."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x)
