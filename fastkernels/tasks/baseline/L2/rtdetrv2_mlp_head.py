"""RTDetrV2 MLP prediction head."""

from __future__ import annotations

import torch.nn as nn

from ..L1.linear import Linear
from ..L1.relu import ReLU


class RTDetrV2MLPPredictionHead(nn.Module):
    def __init__(self, config, input_dim, d_model, output_dim, num_layers):
        super().__init__()
        del config
        self.num_layers = num_layers
        h = [d_model] * (num_layers - 1)
        self.layers = nn.ModuleList(Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self._relu = ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self._relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
