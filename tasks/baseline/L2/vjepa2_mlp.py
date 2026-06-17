"""V-JEPA 2 MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.linear import Linear


class VJEPA2MLP(nn.Module):
    def __init__(self, config, hidden_size: int = 1024, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        approximate = "tanh" if getattr(config, "hidden_act", "gelu") == "gelu_new" else "none"
        self.fc1 = Linear(hidden_size, hidden_features, bias=True)
        self.activation = GELU(approximate=approximate)
        self.fc2 = Linear(hidden_features, hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state
