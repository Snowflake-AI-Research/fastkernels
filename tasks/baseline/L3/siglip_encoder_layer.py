"""SigLIP vision encoder layer (L3 composite).

Pre-norm transformer block: LayerNorm -> Attention -> residual ->
LayerNorm -> MLP -> residual.

Mirrors HuggingFace Transformers ``SiglipEncoderLayer``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.siglip_attention import SigLIPAttention
from ..L2.siglip_mlp import SigLIPMLP


class SigLIPEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, intermediate_size: int,
                 layer_norm_eps: float = 1e-6):
        super().__init__()
        self.layer_norm1 = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.self_attn = SigLIPAttention(embed_dim, num_heads)
        self.layer_norm2 = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = SigLIPMLP(embed_dim, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
