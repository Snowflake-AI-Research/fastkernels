"""SigLIP vision encoder attention (L2 composite).

Dense multi-head attention with no positional encoding (position information
comes from learned position embeddings added at the patch embedding stage).

Mirrors HuggingFace Transformers ``SiglipAttention``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear


class SigLIPAttention(nn.Module):
    """Multi-head attention for SigLIP vision encoder.

    All heads are attention heads (no GQA). Non-causal.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        self.attn = DenseAttention()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        scale = self.head_dim ** -0.5
        out = self.attn(q, k, v, softmax_scale=scale, causal=False)

        out = out.reshape(bsz, seq_len, -1)
        return self.out_proj(out)
