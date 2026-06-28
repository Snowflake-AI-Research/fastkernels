"""GEGLU activation: gated linear unit with GELU gate.

Linear(dim, inner_dim * 2) -> split -> GELU(gate) * hidden.
Mirrors diffusers' GEGLU class.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.linear import Linear


class GEGLU(nn.Module):
    """GEGLU activation with a fused linear projection.

    Projects input to 2x inner_dim, splits, applies GELU to gate half,
    and multiplies with the hidden half.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = Linear(dim_in, dim_out * 2, bias=True)
        self.gelu = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)
