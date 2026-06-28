"""Shared MLP for recommender models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.relu import ReLU


class RecsysMLP(nn.Module):
    def __init__(self, layer_dims: list[int], activate_last: bool = False):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions")
        self.layers = nn.ModuleList([
            Linear(layer_dims[i], layer_dims[i + 1], bias=True)
            for i in range(len(layer_dims) - 1)
        ])
        self.activation = ReLU()
        self.activate_last = activate_last

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            is_last = index == len(self.layers) - 1
            if self.activate_last or not is_last:
                hidden_states = self.activation(hidden_states)
        return hidden_states
