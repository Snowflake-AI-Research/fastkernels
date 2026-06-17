"""T5 encoder block: self-attention + FFN with pre-norm residuals (L3).

T5LayerSelfAttention: T5LayerNorm -> T5SelfAttention -> residual add.
T5LayerFF: T5LayerNorm -> T5Dense{Gated}ActDense -> residual add.
T5Block: T5LayerSelfAttention + T5LayerFF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import T5Config

from ..L1.t5_layer_norm import T5LayerNorm
from ..L2.t5_attention import T5SelfAttention
from ..L2.t5_dense import T5DenseActDense, T5DenseGatedActDense


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, mask=mask, position_bias=position_bias,
        )
        hidden_states = hidden_states + attn_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + ff_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias
