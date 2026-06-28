"""Downsample2D for SDXL UNet — stride-2 Conv2d spatial downsampling.

Mirrors diffusers' Downsample2D with use_conv=True.
Parameter name: conv (Conv2d).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d


class Downsample2D(nn.Module):
    """Spatial downsampling via stride-2 convolution."""

    def __init__(self, channels: int, out_channels: int | None = None, padding: int = 1):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=padding)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)
