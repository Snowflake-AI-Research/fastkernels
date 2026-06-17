"""SigLIP vision encoder MLP (L2 composite).

Two-layer MLP with GELU activation. No gating.

Mirrors HuggingFace Transformers ``SiglipMLP``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.linear import Linear


class SigLIPMLP(nn.Module):
    def __init__(self, embed_dim: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(embed_dim, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, embed_dim, bias=True)
        self.act = GELU(approximate="none")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))
