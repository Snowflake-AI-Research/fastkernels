"""V-JEPA 2 attentive pooler."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.vjepa2_attention import VJEPA2PoolerCrossAttention, VJEPA2PoolerSelfAttention
from ..L2.vjepa2_mlp import VJEPA2MLP


class VJEPA2PoolerSelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VJEPA2PoolerSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size, mlp_ratio=config.mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class VJEPA2PoolerCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = VJEPA2PoolerCrossAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size, mlp_ratio=config.mlp_ratio)

    def forward(
        self,
        queries: torch.Tensor,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = queries
        hidden_state = self.layer_norm1(hidden_state)
        hidden_state, *attn_weights = self.cross_attn(
            queries,
            hidden_state,
            hidden_state,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)
        if output_attentions:
            outputs += tuple(attn_weights)
        return outputs


class VJEPA2AttentivePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config)
        self.self_attention_layers = nn.ModuleList([
            VJEPA2PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)
        ])

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state, attention_mask=None)[0]
        queries = self.query_tokens.repeat(hidden_state.shape[0], 1, 1)
        hidden_state = self.cross_attention_layer(queries, hidden_state)[0]
        return hidden_state.squeeze(1)
