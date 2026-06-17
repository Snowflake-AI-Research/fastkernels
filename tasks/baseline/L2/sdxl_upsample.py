"""Upsample2D for SDXL UNet — nearest-neighbor 2x interpolation + Conv2d.

Mirrors diffusers' Upsample2D with use_conv=True.
Parameter name: conv (Conv2d).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.conv2d import Conv2d


class Upsample2D(nn.Module):
    """Spatial upsampling via 2x nearest interpolation + convolution."""

    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor, output_size: int | None = None) -> torch.Tensor:
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states
