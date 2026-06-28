"""DP3 1-D conv block: Conv1d -> GroupNorm -> Mish.

Mirrors ``diffusion_policy_3d.model.diffusion.conv1d_components.Conv1dBlock``
exactly, including the same ``block`` Sequential layout so checkpoint keys
load without remapping.  L2 composes only L1 ops per CLAUDE.md: the
1-D conv ops are :class:`L1.conv1d_native.Conv1dNative` and
:class:`L1.conv_transpose1d.ConvTranspose1d`, both of which expose
``weight`` and ``bias`` as direct ``nn.Parameter`` attributes (matching
the reference's ``nn.Conv1d`` / ``nn.ConvTranspose1d`` parameter naming
under ``block.0.weight`` etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv1d_native import Conv1dNative
from ..L1.conv_transpose1d import ConvTranspose1d
from ..L1.group_norm import GroupNorm
from ..L1.mish import Mish


class Conv1dBlock(nn.Module):
    """Conv1d (kernel_size, padding=ks//2) -> GroupNorm(n_groups) -> Mish."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Conv1dNative(
                inp_channels, out_channels, kernel_size,
                padding=kernel_size // 2,
            ),
            # eps=1e-5 matches torch.nn.GroupNorm default (the reference
            # uses nn.GroupNorm directly); fastkernels's L1 GroupNorm defaults
            # to eps=1e-6.
            GroupNorm(n_groups, out_channels, eps=1e-5),
            Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    """Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = Conv1dNative(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
