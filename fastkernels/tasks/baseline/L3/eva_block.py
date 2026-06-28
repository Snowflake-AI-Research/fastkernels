"""EVA pre-norm transformer block with layer-scale (L3).

Pre-normalization block with optional layer-scale:
  x = x + gamma1 * attn(norm1(x), rope)
  x = x + gamma2 * mlp(norm2(x))

Used by DINOv3 and other EVA-family architectures.

Reference: timm/models/eva.py EvaBlock
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.eva_attention import EvaAttention
from ..L2.swiglu_mlp import SwiGLUMlp


class EvaBlock(nn.Module):
    """EVA transformer block with layer-scale and SwiGLU MLP.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dim expansion ratio.
        qkv_bias: Bias in QKV projection.
        qkv_fused: Use fused QKV.
        num_prefix_tokens: Cls/register tokens exempt from RoPE.
        rotate_half: Use rotate-half RoPE layout.
        init_values: Layer-scale initial value (None disables).
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
        swiglu_align_to: Alignment multiple for SwiGLU hidden dim.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qkv_fused: bool = True,
        num_prefix_tokens: int = 1,
        rotate_half: bool = True,
        init_values: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        swiglu_align_to: int = 0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rotate_half=rotate_half,
        )
        self.gamma_1 = nn.Parameter(torch.full((dim,), init_values)) if init_values is not None else None
        self.norm2 = LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = SwiGLUMlp(
            in_features=dim,
            hidden_features=hidden_features,
            bias=True,
            drop=proj_drop,
            align_to=swiglu_align_to,
        )
        self.gamma_2 = nn.Parameter(torch.full((dim,), init_values)) if init_values is not None else None

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x
