"""EVA-style attention with RoPE (L2).

Multi-head attention with rotary position embeddings, optional
no-k-bias, q/k normalization, and scale normalization. Used by
DINOv3 and other EVA-family models.

Reference: timm/models/eva.py EvaAttention
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.layer_norm import LayerNorm
from ..L1.dinov3_rope import apply_rot_embed_cat


class EvaAttention(nn.Module):
    """EVA attention with RoPE and optional biases.

    DINOv3 7B config: qkv_bias=False, qkv_fused=True, rotate_half=True,
    no q/k norm, no scale norm.

    Args:
        dim: Input embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: Bias in QKV projections.
        qkv_fused: Use single fused QKV projection.
        num_prefix_tokens: Number of cls/register tokens exempt from RoPE.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
        rotate_half: Use rotate-half RoPE layout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qkv_fused: bool = True,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rotate_half: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.rotate_half = rotate_half

        if qkv_fused:
            self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv = None
            self.q_proj = Linear(dim, dim, bias=qkv_bias)
            self.k_proj = Linear(dim, dim, bias=qkv_bias)
            self.v_proj = Linear(dim, dim, bias=qkv_bias)

        self.proj = Linear(dim, dim, bias=True)
        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([
                q[:, :, :npt, :],
                apply_rot_embed_cat(q[:, :, npt:, :], rope, half=self.rotate_half),
            ], dim=2).type_as(v)
            k = torch.cat([
                k[:, :, :npt, :],
                apply_rot_embed_cat(k[:, :, npt:, :], rope, half=self.rotate_half),
            ], dim=2).type_as(v)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
