"""Conv3D patch embedding for vision encoder."""

from __future__ import annotations

import torch.nn as nn


class Conv3d(nn.Module):
    """Conv3D wrapper matching vllm's Conv3dLayer interface."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple[int, ...], stride: tuple[int, ...] | None = None,
                 bias: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride or kernel_size, bias=bias)
    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        return self.conv(x)

