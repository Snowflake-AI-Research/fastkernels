"""SwiGLU MLP (L2).

Gated MLP with SiLU activation: out = fc2(SiLU(fc1_g(x)) * fc1_x(x)).
Separate gate and value linear projections with optional alignment
padding for hardware efficiency.

Reference: timm/layers/mlp.py SwiGLU
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.silu import SiLU


class SwiGLUMlp(nn.Module):
    """SwiGLU MLP with separate gate and value projections.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension before gating.
        out_features: Output dimension (defaults to in_features).
        bias: Use bias in linear layers.
        drop: Dropout rate.
        align_to: Align hidden_features to this multiple for HW efficiency.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        drop: float = 0.0,
        align_to: int = 0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        if align_to > 0:
            hidden_features = ((hidden_features + align_to - 1) // align_to) * align_to

        self.fc1_g = Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = Linear(in_features, hidden_features, bias=bias)
        self.act = SiLU()
        self.fc2 = Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = self.act(self.fc1_g(x))
        x = x_gate * self.fc1_x(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
