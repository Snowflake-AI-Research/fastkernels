"""CLIP encoder layer: pre-norm attention + MLP with residual connections (L3).

Standard transformer encoder layer used in CLIP text models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPTextConfig

from ..L1.layer_norm import LayerNorm
from ..L2.clip_attention import CLIPAttention
from ..L2.clip_mlp import CLIPMLP


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
