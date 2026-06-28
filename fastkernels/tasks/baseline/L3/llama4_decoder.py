"""Llama 4 decoder layer: attention + MoE/MLP with NoPE support.

MoE vs dense MLP is selected per layer via ``interleave_moe_layer_step``
(matching vLLM's ``Llama4DecoderLayer``).

Weight names match checkpoint:
  layers.{i}.self_attn.{q,k,v,o}_proj.weight
  layers.{i}.feed_forward.*
  layers.{i}.input_layernorm.weight
  layers.{i}.post_attention_layernorm.weight
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.llama4_moe import Llama4MoE
from ..L2.llama_mlp import LlamaMLP


@dataclass
class _DenseMlpConfig:
    """Minimal config for LlamaMLP on dense (non-MoE) layers."""
    hidden_size: int
    intermediate_size: int


class Llama4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, rotary_emb: nn.Module | None = None):
        super().__init__()
        no_rope_layers = getattr(config, "no_rope_layers", None) or []
        nope = no_rope_layers[layer_idx] == 0 if layer_idx < len(no_rope_layers) else False

        chunk_size = getattr(config, "attention_chunk_size", None)
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            nope=nope,
            use_weightless_qk_norm=getattr(config, "use_qk_norm", False),
            attn_temperature_tuning=getattr(config, "attn_temperature_tuning", False),
            floor_scale=getattr(config, "floor_scale", 8192.0),
            attn_scale=getattr(config, "attn_scale", 0.1),
            rms_norm_eps=config.rms_norm_eps,
            attention_chunk_size=chunk_size if not nope else None,
        )

        step = getattr(config, "interleave_moe_layer_step", 1)
        is_moe_layer = step > 0 and (layer_idx + 1) % step == 0
        if is_moe_layer:
            self.feed_forward = Llama4MoE(config)
        else:
            mlp_cfg = _DenseMlpConfig(
                hidden_size=config.hidden_size,
                intermediate_size=getattr(config, "intermediate_size_mlp", config.intermediate_size),
            )
            self.feed_forward = LlamaMLP(mlp_cfg)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual
