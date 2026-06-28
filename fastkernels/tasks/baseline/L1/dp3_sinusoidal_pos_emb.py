"""Sinusoidal position embedding for DP3 diffusion timestep encoding.

Mirrors ``diffusion_policy_3d.model.diffusion.positional_embedding.SinusoidalPosEmb``.
Distinct from fastkernels's existing ``sinusoidal_embed.SinusoidalEmbed`` which
exposes configurable ``min_period`` / ``max_period`` for Pi0 flow-matching;
DP3 uses the standard ``log(10000)`` schedule.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DP3SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding with the standard log(10000) frequency schedule.

    Args:
        dim: output embedding dimension (must be even).
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) timestep tensor (long or float).
        Returns:
            (B, dim) sinusoidal embedding.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
