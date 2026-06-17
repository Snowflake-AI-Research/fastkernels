"""Qwen3-Next decoder layer: hybrid GDN/full attention + MoE.

Dispatches to GDN linear attention or full attention based on layer type.
All layers use MoE (every layer is sparse in Qwen3-Next).
Uses GemmaRMSNorm (weight + 1 convention).
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention
from ..L2.qwen3_next_attention import Qwen3NextAttention
from ..L2.shared_expert_moe import SharedExpertMoE


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                layer_idx=layer_idx,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
            )
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")

        # MoE for all Qwen3-Next layers (every layer is sparse).
        self.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            routing="softmax",
            correction_bias=False,
            renormalize=config.norm_topk_prob,
            routed_scaling_factor=1.0,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            shared_expert_attr_name="shared_expert",
            shared_expert_gate=True,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, residual, positions=None,
                rotary_emb=None, state_manager=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Attention
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, state_manager=state_manager,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, rotary_emb=rotary_emb, positions=positions,
                state_manager=state_manager,
            )

        # Post-attention norm + MLP
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
