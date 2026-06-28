"""Multi-head attention for SDXL UNet spatial transformers.

Supports both self-attention and cross-attention with separate Q/K/V
projections.  Uses DenseAttention with the ``sdpa`` backend to match
diffusers' AttnProcessor2_0 numerics.

Parameter names match diffusers' Attention class:
  to_q, to_k, to_v, to_out.0 (Linear + bias).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear


class SDXLAttention(nn.Module):
    """Multi-head attention with Q/K/V projections.

    Args:
        query_dim: Dimension of the query input.
        cross_attention_dim: Dimension of encoder_hidden_states for cross-attn.
            If None, self-attention is used.
        heads: Number of attention heads.
        dim_head: Dimension per head.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim or query_dim

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = Linear(query_dim, inner_dim, bias=False)
        self.to_k = Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([
            Linear(inner_dim, query_dim, bias=True),
        ])

        self.attn = DenseAttention(backend="sdpa")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        q = q.view(batch_size, -1, self.heads, self.dim_head)
        k = k.view(batch_size, -1, self.heads, self.dim_head)
        v = v.view(batch_size, -1, self.heads, self.dim_head)

        out = self.attn(q, k, v, softmax_scale=self.scale, causal=False)

        out = out.reshape(batch_size, -1, self.heads * self.dim_head)
        out = out.to(q.dtype)
        out = self.to_out[0](out)
        return out
