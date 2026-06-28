"""Standard ViT multi-head self-attention (L2).

Fused QKV projection, optional QK normalization, SDPA backend.
Used by SigLIP-2 (NaFlexVit) and other standard ViT architectures.

Reference: timm/layers/attention.py Attention
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.layer_norm import LayerNorm


class VitEncoderAttention(nn.Module):
    """Standard multi-head self-attention with fused QKV.

    Args:
        dim: Input embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: Use bias in QKV projection.
        proj_bias: Use bias in output projection.
        attn_drop: Dropout on attention weights.
        proj_drop: Dropout after output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=proj_bias)
        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
