"""RTDetrV2 Conv-Norm-Activation layer."""

from __future__ import annotations

import torch.nn as nn

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.gelu import GELU
from ..L1.relu import ReLU
from ..L1.silu import SiLU

_ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}


def _get_activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    return _ACTIVATIONS[name.lower()]()


class RTDetrV2ConvNormLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        self.activation = _get_activation(activation)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state
