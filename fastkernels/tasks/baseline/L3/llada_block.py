"""LLaDA transformer block."""

from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.llada_attention import LLaDAAttention
from ..L2.llada_mlp import LLaDAMLP


class LLaDABlock(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ff_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = LLaDAAttention(
            hidden_size=config.d_model,
            num_attention_heads=config.n_heads,
            num_key_value_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            bias=config.include_bias or config.include_qkv_bias,
            rope_full_precision=config.rope_full_precision,
        )
        self.mlp = LLaDAMLP(
            hidden_size=config.d_model,
            intermediate_size=config.mlp_hidden_size,
            bias=config.include_bias,
        )

    def forward(
        self,
        hidden_states,
        attention_bias=None,
        layer_past=None,
        use_cache: bool = False,
        replace_position=None,
    ):
        attn, cache = self.attention(
            self.attn_norm(hidden_states),
            attention_bias=attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            replace_position=replace_position,
        )
        hidden_states = hidden_states + attn
        hidden_states = hidden_states + self.mlp(self.ff_norm(hidden_states))
        return hidden_states, cache
