"""ConvTranspose1d parametric op wrapping F.conv_transpose1d.

Used by DP3's 1-D U-Net upsampling block.  Stores ``weight`` and
``bias`` as direct ``nn.Parameter`` attributes so reference state_dicts
load with no remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTranspose1d(nn.Module):
    """Parametric 1-D transposed convolution.  Forward dispatches to
    ``F.conv_transpose1d``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        # ConvTranspose weight layout: [in_channels, out_channels // groups, kernel_size]
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
