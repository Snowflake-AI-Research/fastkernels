"""Attention pooling with latent query (L2).

Cross-attention pooling: a learnable latent query attends to sequence tokens
to produce a fixed-size pooled representation. Used as MAP (multi-head
attention pooling) in SigLIP-2.

Reference: timm/layers/attention_pool.py AttentionPoolLatent
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.layer_norm import LayerNorm


class AttentionPoolLatent(nn.Module):
    """Attention pooling with a learnable latent query.

    Args:
        in_features: Input token dimension.
        out_features: Output dimension (0 disables proj/norm/mlp).
        embed_dim: Internal attention dimension (defaults to in_features).
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Bias in Q/KV projections.
        latent_len: Number of latent query tokens.
        pool_type: How to reduce latent_len > 1 ("token" or "avg").
        drop: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        embed_dim: int | None = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        latent_len: int = 1,
        pool_type: str = "token",
        drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        if out_features is None:
            out_features = in_features
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool = pool_type
        self.latent_len = latent_len

        self.latent = nn.Parameter(torch.zeros(1, latent_len, embed_dim))
        self.q = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = Linear(embed_dim, embed_dim * 2, bias=qkv_bias)

        if out_features > 0:
            self.proj = Linear(embed_dim, out_features, bias=True)
            self.proj_drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
            self.norm = LayerNorm(out_features)
            self.mlp = _PoolMlp(out_features, int(out_features * mlp_ratio), out_features)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
            self.norm = nn.Identity()
            self.mlp = None
            out_features = embed_dim

        self.out_features = out_features

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.mlp is not None:
            x = x + self.mlp(self.norm(x))

        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(1)
        return x


class _PoolMlp(nn.Module):
    """Simple MLP used inside AttentionPoolLatent."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.fc2 = Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
