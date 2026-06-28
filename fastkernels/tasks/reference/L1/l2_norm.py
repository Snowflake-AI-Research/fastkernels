"""L2 normalization along a single dimension: x / ||x||_2.

Wraps F.normalize(p=2). Used by RWKV7 to L2-normalize per-head keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)
