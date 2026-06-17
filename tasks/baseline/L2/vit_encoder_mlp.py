"""Standard ViT MLP (L2).

Two-layer feed-forward network: fc1 -> act -> drop -> fc2 -> drop.
Configurable activation (GELU, GELU-tanh, etc.).
Used by SigLIP-2 (NaFlexVit) and other standard ViT architectures.

Reference: timm/layers/mlp.py Mlp
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.gelu import GELU


class VitEncoderMlp(nn.Module):
    """Standard two-layer MLP.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (defaults to in_features).
        out_features: Output dimension (defaults to in_features).
        act_approximate: GELU approximation mode ("none" or "tanh").
        bias: Use bias in linear layers.
        drop: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_approximate: str = "none",
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = Linear(in_features, hidden_features, bias=bias)
        self.act = GELU(approximate=act_approximate)
        self.fc2 = Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
