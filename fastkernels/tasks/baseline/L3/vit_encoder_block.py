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
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        # promote_fp32=False and eps=1e-6 to match timm, whose ViT Block uses
        # ``timm.layers.LayerNorm`` -- an nn.LayerNorm subclass that calls plain
        # ``F.layer_norm`` in the model dtype, with eps defaulting to 1e-6 (not
        # torch's 1e-5). Promoting costs an x.float() before and a cast back
        # after every norm: at 27 blocks x 2 norms, bs=32 and 576 patches, that
        # is ~340 MB of extra traffic per site, ~18 GB per forward.
        self.norm1 = LayerNorm(dim, eps=norm_eps, promote_fp32=False)
        self.attn = VitEncoderAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = LayerNorm(dim, eps=norm_eps, promote_fp32=False)
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
