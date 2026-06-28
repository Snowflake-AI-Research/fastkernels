"""DP3 conditional residual block (FiLM-conditioned).

Mirrors ``diffusion_policy_3d.model.diffusion.conditional_unet1d.ConditionalResidualBlock1D``
restricted to ``condition_type="film"`` — the only mode used by the released
DP3 / Simple-DP3 configs.

Layout:
    blocks.0: Conv1dBlock(in -> out)
    blocks.1: Conv1dBlock(out -> out)
    cond_encoder: Mish -> Linear(cond_dim, 2*out) -> reshape (FiLM scale+bias)
    residual_conv: Conv1d(in, out, 1)  if in != out, else Identity

Forward:
    out = blocks.0(x)
    embed = cond_encoder(cond)            # (B, 2*out, 1)
    scale, bias = split(embed)
    out = scale * out + bias
    out = blocks.1(out)
    out = out + residual_conv(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.mish import Mish

from .dp3_conv1d_block import Conv1dBlock


class _RearrangeBT1(nn.Module):
    """``b t -> b t 1`` — append a length-1 horizon axis to the FiLM embed.

    Used inside ``cond_encoder`` Sequential so that the state_dict key layout
    matches the reference's ``Rearrange('batch t -> batch t 1')`` element.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


class ConditionalResidualBlock1D(nn.Module):
    """Two-Conv1dBlock residual with FiLM (scale+bias) conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            Mish(),
            Linear(cond_dim, cond_channels),
            _RearrangeBT1(),
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0]
        bias = embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
