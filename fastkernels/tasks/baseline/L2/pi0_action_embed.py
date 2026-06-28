"""Pi0 action-time embedding (L2 composite).

Embeds robot state, noisy actions, and flow-matching timestep into the
action expert's hidden space. The state becomes one token; each noisy action
step becomes one token conditioned on the flow timestep via a two-layer MLP
with SiLU activation.

Mirrors HuggingFace Transformers ``PI0ActionTimeEmbedding``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU
from ..L1.sinusoidal_embed import SinusoidalEmbed


class Pi0ActionTimeEmbedding(nn.Module):
    """Embed state + noisy actions + flow timestep for the Pi0 action expert.

    Args:
        expert_hidden_size: Hidden dimension of the action expert.
        max_action_dim: Maximum action vector dimension (zero-padded).
        max_state_dim: Maximum state vector dimension (zero-padded).
        min_period: Min period for sinusoidal time embedding.
        max_period: Max period for sinusoidal time embedding.
    """

    def __init__(
        self,
        expert_hidden_size: int,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        min_period: float = 0.004,
        max_period: float = 4.0,
    ):
        super().__init__()
        self.sinusoid_embeds = SinusoidalEmbed(
            expert_hidden_size, min_period=min_period, max_period=max_period,
        )
        self.action_in_proj = Linear(max_action_dim, expert_hidden_size)
        self.state_proj = Linear(max_state_dim, expert_hidden_size)
        self.action_time_mlp_in = Linear(2 * expert_hidden_size, expert_hidden_size)
        self.action_time_mlp_out = Linear(expert_hidden_size, expert_hidden_size)
        self.act = SiLU()

    def forward(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state: (batch, max_state_dim) robot proprioceptive state.
            noise: (batch, chunk_size, max_action_dim) noisy action chunk.
            timestep: (batch,) flow-matching timestep in [0, 1].

        Returns:
            (batch, 1 + chunk_size, expert_hidden_size) — state token
            followed by action tokens.
        """
        state_embeds = self.state_proj(state)
        action_embeds = self.action_in_proj(noise)

        time_embeds = self.sinusoid_embeds(timestep)
        time_embeds = time_embeds[:, None, :].expand_as(action_embeds).to(
            dtype=action_embeds.dtype,
        )

        action_time = torch.cat([action_embeds, time_embeds], dim=2)
        action_time = self.action_time_mlp_out(self.act(self.action_time_mlp_in(action_time)))

        return torch.cat([state_embeds[:, None, :], action_time], dim=1)
