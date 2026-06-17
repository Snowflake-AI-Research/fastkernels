"""Mixtral decoder layer: attention + MoE with RMSNorm residual connections."""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.mixtral_moe import MixtralMoE


class MixtralDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
        )
        self.block_sparse_moe = MixtralMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual
