"""Conv1d parametric op wrapping F.conv1d.

Mirrors :class:`L1.conv2d.Conv2d` for the 1-D case.  Stores ``weight`` and
``bias`` as direct ``nn.Parameter`` attributes (not nested under
``self.conv`` like the older :class:`L1.conv1d.Conv1d` Whisper wrapper),
so a state_dict from a reference model that uses ``nn.Conv1d`` directly
loads with no parameter-name remapping (``block.0.weight`` /
``block.0.bias`` etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dNative(nn.Module):
    """Parametric 1-D convolution.  Forward dispatches to ``F.conv1d``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
