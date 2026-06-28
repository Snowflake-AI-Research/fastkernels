"""Sinusoidal time/position embedding for flow matching models.

Computes sin/cos embeddings from a scalar timestep using log-spaced
frequencies controlled by min_period and max_period. Used by Pi0's
action-time conditioning.

Mirrors HuggingFace Transformers ``PI0TimestepEmbeddings``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbed(nn.Module):
    """Sinusoidal embedding with configurable frequency range.

    Args:
        embed_dim: Output embedding dimension (must be even).
        min_period: Minimum oscillation period.
        max_period: Maximum oscillation period.
    """

    def __init__(self, embed_dim: int, min_period: float = 0.004,
                 max_period: float = 4.0):
        super().__init__()
        fraction = torch.linspace(0.0, 1.0, embed_dim // 2, dtype=torch.float32)
        period = min_period * (max_period / min_period) ** fraction
        sinusoid_freq = 1.0 / period * 2 * math.pi
        self.register_buffer("sinusoid_freq", sinusoid_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar timesteps.

        Args:
            t: (batch,) float tensor of timestep values.

        Returns:
            (batch, embed_dim) sinusoidal embedding.
        """
        emb = self.sinusoid_freq[None, :].float() * t[:, None].float()
        return torch.cat([emb.sin(), emb.cos()], dim=1)
