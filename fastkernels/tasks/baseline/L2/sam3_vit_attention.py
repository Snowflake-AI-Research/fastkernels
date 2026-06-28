"""ViT-Det windowed/global multi-head attention for SAM3.

Handles QKV projection, optional 2D RoPE, windowed partitioning for local
attention, and output projection. Supports both 4-D (B, H, W, C) and
3-D (B, L, C) input layouts.

Reference: sam3/model/vitdet.py Attention
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.sam3_rope import Sam3RoPE2D


class Sam3ViTAttention(nn.Module):
    """Multi-head attention for SAM3 ViT with optional 2D RoPE.

    Args:
        dim: Input/output channel dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to add bias to QKV projection.
        use_rope: Whether to apply 2D rotary position embedding.
        input_size: Spatial resolution (H, W) for RoPE precomputation.
        rope_theta: Base frequency for RoPE.
        rope_pt_size: Pre-training resolution for RoPE tiling/interpolation.
        rope_tiled: Tile RoPE from rope_pt_size instead of interpolating.
        rope_interp: Interpolate RoPE to input_size.
        cls_token: Whether a CLS token is present (prepended to sequence).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rope: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        rope_theta: float = 10000.0,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        cls_token: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.cls_token = cls_token

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.attn = DenseAttention()

        self.rope: Sam3RoPE2D | None = None
        if use_rope and input_size is not None:
            self.rope = Sam3RoPE2D(
                head_dim=self.head_dim,
                input_size=input_size,
                theta=rope_theta,
                tiled=rope_tiled,
                pt_size=rope_pt_size if rope_pt_size is not None else input_size,
                interp=rope_interp,
                cls_token=cls_token,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            B, H, W, _ = x.shape
            L = H * W
        else:
            B, L, _ = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, heads, L, head_dim)

        if self.rope is not None:
            q, k = self.rope(q, k)

        # DenseAttention expects (B, L, heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v)  # (B, L, heads, head_dim)

        if x.ndim == 4:
            out = out.reshape(B, H, W, -1)
        else:
            out = out.reshape(B, L, -1)

        return self.proj(out)
