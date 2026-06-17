"""ViT-Det transformer block for SAM3.

A single ViT block with windowed or global attention, pre-norm residuals,
and optional drop-path. Handles window partitioning/unpartitioning internally.

Reference: sam3/model/vitdet.py Block
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L2.sam3_vit_attention import Sam3ViTAttention
from ..L2.sam3_vit_mlp import Sam3ViTMLP


def _window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition (B, H, W, C) into (B*nW, ws, ws, C) with padding."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def _window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Reverse window partition and remove padding."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class Sam3ViTBlock(nn.Module):
    """Single ViT-Det block with optional windowed attention.

    Args:
        dim: Channel dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Bias in QKV projection.
        drop_path: Stochastic depth rate.
        window_size: Window size for local attention (0 = global).
        use_rope: Enable 2D RoPE.
        input_size: Full spatial resolution (H, W).
        rope_pt_size: Pre-training resolution for RoPE.
        rope_tiled: Tile RoPE frequencies.
        rope_interp: Interpolate RoPE frequencies.
        cls_token: CLS token present in sequence.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        window_size: int = 0,
        use_rope: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        cls_token: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size

        attn_input_size = (
            (window_size, window_size) if window_size > 0 else input_size
        )

        self.norm1 = LayerNorm(dim)
        self.attn = Sam3ViTAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            input_size=attn_input_size,
            rope_pt_size=rope_pt_size,
            rope_tiled=rope_tiled,
            rope_interp=rope_interp,
            cls_token=cls_token,
        )

        self.norm2 = LayerNorm(dim)
        self.mlp = Sam3ViTMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=dropout,
        )

        self.drop_path = nn.Identity()
        if drop_path > 0.0:
            try:
                from timm.layers import DropPath
                self.drop_path = DropPath(drop_path)
            except ImportError:
                pass

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = _window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.mlp(self.norm2(x))))
        return x
