"""ViT MLP block for SAM3.

Two-layer feed-forward network with GELU activation, matching the Mlp class
in vitdet.py. Dropout is supported but defaults to 0.

Reference: sam3/model/vitdet.py Mlp
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.linear import Linear


class Sam3ViTMLP(nn.Module):
    """Two-layer MLP with GELU activation for SAM3 ViT blocks.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (defaults to in_features).
        out_features: Output dimension (defaults to in_features).
        drop: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
