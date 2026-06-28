"""Text encoder residual attention block for SAM3.

Self-attention + MLP with pre-norm residual connections, matching the
ResidualAttentionBlock used inside VETextEncoder's TextTransformer.

Reference: sam3/model/text_encoder_ve.py ResidualAttentionBlock
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


class Sam3TextAttentionBlock(nn.Module):
    """Residual attention block for SAM3 text encoder.

    Pre-norm architecture: LN -> self-attn -> residual -> LN -> MLP -> residual.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        mlp_ratio: Expansion ratio for the MLP hidden dimension.
    """

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)

        self.q_proj = Linear(d_model, d_model, bias=True)
        self.k_proj = Linear(d_model, d_model, bias=True)
        self.v_proj = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)
        self.attn = DenseAttention()

        mlp_width = int(d_model * mlp_ratio)
        self.mlp_fc1 = Linear(d_model, mlp_width, bias=True)
        self.mlp_act = GELU()
        self.mlp_fc2 = Linear(mlp_width, d_model, bias=True)

    def _self_attention(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_head, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_head, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_head, self.head_dim)

        out = self.attn(q, k, v, causal=attn_mask is not None)
        return self.out_proj(out.reshape(B, L, -1))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self._self_attention(self.ln_1(x), attn_mask)
        h = self.ln_2(x)
        h = self.mlp_fc2(self.mlp_act(self.mlp_fc1(h)))
        return x + h
