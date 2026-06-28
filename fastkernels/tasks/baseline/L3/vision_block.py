"""Vision transformer block for Qwen VL models.

Unified across Qwen2-VL and Qwen3-VL:
  - act_fn: Qwen2 uses QuickGELU (default), Qwen3 uses SiLU.
  - norm_eps: configurable LayerNorm epsilon.

Uses LayerNorm (not RMSNorm) with pre-norm residual connections,
encoder-only attention, and vision MLP.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.quickgelu import QuickGELU
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import VisionMLP


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), cu_seqlens,
            rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x
