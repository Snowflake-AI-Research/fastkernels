"""RTDetrV2 CSP RepVGG layer."""

from __future__ import annotations

import torch.nn as nn

from .rtdetrv2_conv_norm import RTDetrV2ConvNormLayer
from .rtdetrv2_repvgg_block import RTDetrV2RepVggBlock


class RTDetrV2CSPRepLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        hidden_channels = int(out_channels * config.hidden_expansion)
        activation = config.activation_function
        self.conv1 = RTDetrV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = RTDetrV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[RTDetrV2RepVggBlock(config) for _ in range(3)])
        if hidden_channels != out_channels:
            self.conv3 = RTDetrV2ConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)
