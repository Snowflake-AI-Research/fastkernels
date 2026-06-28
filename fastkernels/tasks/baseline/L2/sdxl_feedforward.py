"""FeedForward network with GEGLU activation for SDXL.

GEGLU(dim, inner_dim) -> Linear(inner_dim, dim_out).
Mirrors diffusers' FeedForward with activation_fn="geglu".

Parameter names match diffusers: net.0 (GEGLU), net.2 (output Linear).
The nn.ModuleList with index gaps (0, 2) mirrors the diffusers convention
where index 1 is a dropout layer (omitted for inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from .geglu import GEGLU


class FeedForward(nn.Module):
    """GEGLU feed-forward network."""

    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.net = nn.ModuleList([
            GEGLU(dim, inner_dim),
            None,
            Linear(inner_dim, dim_out, bias=True),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.net[0](hidden_states)
        hidden_states = self.net[2](hidden_states)
        return hidden_states
