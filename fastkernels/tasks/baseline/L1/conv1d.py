"""Conv1D wrapper for Whisper audio encoder."""

from __future__ import annotations

import torch.nn as nn


class Conv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def stride(self):
        return self.conv.stride

    def forward(self, x):
        return self.conv(x)
