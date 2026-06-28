"""DeepSeek V3.2 decoder layer with MLA attention and MoE.

Layer 0: DeepSeekMLAAttention + dense LlamaMLP
Layers 1+: DeepSeekMLAAttention + DeepSeekMoE
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.deepseek_mla_attention import DeepSeekMLAAttention
from ..L2.deepseek_moe import DeepSeekMoE
from ..L2.llama_mlp import LlamaMLP


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int,
                 rotary_emb: nn.Module,
                 quant_config: dict | None = None,
                 is_v32: bool = False,
                 topk_indices_buffer: torch.Tensor | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)

        self.self_attn = DeepSeekMLAAttention(
            config, rotary_emb=rotary_emb,
            quant_config=quant_config,
            is_v32=is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        moe_layer_freq = getattr(config, 'moe_layer_freq', 1)
        first_k_dense = getattr(config, 'first_k_dense_replace', 1)

        if (config.n_routed_experts is not None
                and layer_idx >= first_k_dense
                and layer_idx % moe_layer_freq == 0):
            self.mlp = DeepSeekMoE(config, quant_config=quant_config)
        else:
            self.mlp = LlamaMLP(config, quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
