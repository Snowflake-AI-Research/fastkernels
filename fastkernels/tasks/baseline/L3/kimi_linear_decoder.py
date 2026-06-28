from __future__ import annotations

import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.kimi_delta_attention import KimiDeltaAttention
from ..L2.kimi_mla_attention import KimiMLAAttention
from ..L2.kimi_moe import KimiMoE
from ..L2.llama_mlp import LlamaMLP


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)

        if self.is_kda:
            self.self_attn = KimiDeltaAttention(
                config,
                layer_idx=layer_idx,
                quant_config=quant_config,
            )
        else:
            self.self_attn = KimiMLAAttention(
                config,
                quant_config=quant_config,
            )

        if config.is_moe_layer(layer_idx):
            self.block_sparse_moe = KimiMoE(config, quant_config=quant_config)
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = LlamaMLP(config, quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states, residual, state_manager=None):
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, state_manager=state_manager)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
