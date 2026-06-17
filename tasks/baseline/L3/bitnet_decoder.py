"""BitNet decoder layer: pre-norm residual with attention and MLP.

Matches HuggingFace's per-layer state-dict layout:

    layers.{i}.input_layernorm.weight
    layers.{i}.self_attn.*
    layers.{i}.post_attention_layernorm.weight
    layers.{i}.mlp.*
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.bitnet_rms_norm import BitNetRMSNorm as RMSNorm
from ..L2.bitnet_attention import BitNetAttention
from ..L2.bitnet_mlp import BitNetMLP


class BitNetDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BitNetAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.mlp = BitNetMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
        )

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor,
                residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
