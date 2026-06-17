"""Gemma4 decoder layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.gemma4_attention import Gemma4Attention
from ..L2.gemma4_mlp import Gemma4MLP
from ..L2.gemma4_moe import Gemma4MoE, Gemma4Router


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, rotary_emb: nn.Module, rotary_dim: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4Attention(
            config,
            layer_idx,
            rotary_emb,
            rotary_dim,
        )
        self.mlp = Gemma4MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4Router(config)
            self.moe = Gemma4MoE(config)
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps,
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps,
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps,
            )

        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, positions, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            router_logits = self.router(residual)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states * self.layer_scalar
