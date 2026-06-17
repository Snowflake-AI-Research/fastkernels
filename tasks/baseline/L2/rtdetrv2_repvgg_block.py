"""RTDetrV2 RepVGG block."""

from __future__ import annotations

import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.relu import ReLU
from ..L1.silu import SiLU
from .rtdetrv2_conv_norm import RTDetrV2ConvNormLayer

_ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}


def _get_activation(name: str) -> nn.Module:
    return _ACTIVATIONS[name.lower()]()


class RTDetrV2RepVggBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = RTDetrV2ConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = RTDetrV2ConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = _get_activation(activation)

    def forward(self, x):
        return self.activation(self.conv1(x) + self.conv2(x))
