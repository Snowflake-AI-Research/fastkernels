"""Standard pre-norm ViT transformer block (L3).

Pre-normalization block: x = x + attn(norm1(x)); x = x + mlp(norm2(x)).
Used by SigLIP-2 (NaFlexVit) and other standard ViT architectures.

Reference: timm/models/vision_transformer.py Block
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.vit_encoder_attention import VitEncoderAttention
from ..L2.vit_encoder_mlp import VitEncoderMlp


class VitEncoderBlock(nn.Module):
    """Pre-norm ViT transformer block.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dim expansion ratio.
        qkv_bias: Bias in QKV projection.
        proj_bias: Bias in output projection.
        act_approximate: GELU approximation mode.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        act_approximate: str = "none",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = VitEncoderAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = VitEncoderMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_approximate=act_approximate,
            bias=proj_bias,
            drop=proj_drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x
