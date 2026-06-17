"""RTDetrV2 transformer encoder layer."""

from __future__ import annotations

import torch.nn as nn

from ..L1.dropout import Dropout
from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.relu import ReLU
from ..L1.silu import SiLU
from .rtdetrv2_multihead_attention import RTDetrV2MultiheadAttention

_ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}


def _get_activation(name: str) -> nn.Module:
    return _ACTIVATIONS[name.lower()]()


class RTDetrV2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.self_attn = RTDetrV2MultiheadAttention(
            embed_dim=config.encoder_hidden_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self._dropout = Dropout(p=config.dropout)
        self._activation_dropout = Dropout(p=config.activation_dropout)
        self.activation_fn = _get_activation(config.encoder_activation_function)
        self.fc1 = Linear(config.encoder_hidden_dim, config.encoder_ffn_dim)
        self.fc2 = Linear(config.encoder_ffn_dim, config.encoder_hidden_dim)
        self.final_layer_norm = LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, output_attentions=False, **kwargs):
        del kwargs
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self._dropout(hidden_states)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self._activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self._dropout(hidden_states)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
