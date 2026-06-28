"""GLA / RetNet decoder layer.

Pre-norm residual:
  attn_norm -> GatedLinearAttention -> residual
  mlp_norm  -> GLAMLP -> residual

Forward signature mirrors FLA's ``GLABlock.forward`` (returns a tuple of
``(hidden_states, attentions, past_key_values)``) so that the same L3
block backs both GLA and RetNet — RetNet just uses ``decay_mode="fixed_per_head"``
and ``use_rotary=True`` in the attention layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.gla_attention import GatedLinearAttention
from ..L2.gla_mlp import GLAMLP


class GLADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedLinearAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            decay_mode=getattr(config, "decay_mode", "learned_low_rank"),
            gate_low_rank_dim=getattr(config, "gate_low_rank_dim", 16),
            gate_logit_normalizer=getattr(config, "gate_logit_normalizer", 16),
            use_rotary=getattr(config, "use_rotary", False),
            rotary_base=getattr(config, "rotary_base", 10000.0),
            rotary_max_position=getattr(config, "max_position_embeddings", 8192),
            norm_eps=config.norm_eps,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        residual = hidden_states
        h = self.attn_norm(
            hidden_states.reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        h, attentions, past_key_values = self.attn(
            hidden_states=h,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + h

        residual = hidden_states
        h = self.mlp_norm(
            hidden_states.reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        hidden_states = residual + self.mlp(h)
        return hidden_states, attentions, past_key_values
