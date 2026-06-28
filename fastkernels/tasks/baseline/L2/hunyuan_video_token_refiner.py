"""HunyuanVideo-1.5 token refiner ops (L2 composites).

Self-attention and feedforward sub-modules used by the L3 token refiner
blocks.  These replicate the behavior of diffusers' ``Attention`` (with
``AttnProcessor2_0``) and ``FeedForward`` (with ``activation_fn="linear-silu"``)
respectively, using only L1 ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.silu import SiLU


class TokenRefinerSelfAttention(nn.Module):
    """Self-attention for the token refiner.

    Equivalent to diffusers' ``Attention`` with ``AttnProcessor2_0`` for
    self-attention (``cross_attention_dim=None``, no qk_norm, no added_kv).

    Weight names (``to_q``, ``to_k``, ``to_v``, ``to_out.0``) match the HF
    checkpoint layout so weight loading works without extra remapping.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        self.to_q = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = Linear(query_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([
            Linear(self.inner_dim, query_dim, bias=True),
            nn.Identity(),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)

        return hidden_states


class _LinearSiLU(nn.Module):
    """Linear projection followed by SiLU activation.

    Matches diffusers' ``LinearActivation(activation="silu")``.
    Parameter name ``self.proj`` matches the HF checkpoint key ``net.0.proj``.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = Linear(dim_in, dim_out, bias=bias)
        self.activation = SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(hidden_states))


class TokenRefinerFeedForward(nn.Module):
    """SiLU feedforward for the token refiner.

    Equivalent to diffusers' ``FeedForward`` with
    ``activation_fn="linear-silu"``: Linear → SiLU → Linear.

    Weight names (``net.0.proj``, ``net.2``) match the HF checkpoint layout.
    """

    def __init__(
        self,
        dim: int,
        mult: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.ModuleList([
            _LinearSiLU(dim, inner_dim),
            nn.Identity(),
            Linear(inner_dim, dim, bias=True),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
