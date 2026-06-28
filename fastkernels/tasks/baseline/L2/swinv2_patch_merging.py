"""SwinV2 patch merging layer (L2).

Merges 2x2 neighboring patches and projects to a higher dimension,
performing spatial 2x downsampling between stages in the hierarchical
Swin Transformer.

Reference: timm/models/swin_transformer_v2.py PatchMerging
"""

from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.layer_norm import LayerNorm


class SwinV2PatchMerging(nn.Module):
    """Patch merging layer for SwinV2 inter-stage downsampling.

    Rearranges 2x2 spatial neighborhoods into the channel dimension,
    then linearly projects from 4*dim to out_dim (default 2*dim).

    Input:  (B, H, W, C) in NHWC format.
    Output: (B, H/2, W/2, out_dim) in NHWC format.

    Args:
        dim: Number of input channels.
        out_dim: Number of output channels (default 2*dim).
    """

    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = Linear(4 * dim, self.out_dim, bias=False)
        self.norm = LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = F.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x
