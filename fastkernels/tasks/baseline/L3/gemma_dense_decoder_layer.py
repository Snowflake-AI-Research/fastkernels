"""Gemma decoder layer with dense attention (L3 composite).

Pre-RMSNorm transformer block for use in Pi0's VLM backbone and action
expert. Uses dense (non-paged) attention with optional KV caching instead
of the paged attention used by LlamaDecoderLayer.

Mirrors HuggingFace Transformers ``GemmaDecoderLayer``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L1.gelu_and_mul import GeluAndMul
from ..L2.gemma_dense_attention import GemmaDenseAttention
from ..L2.parallel_linear import MergedColumnParallelLinear, RowParallelLinear


class GemmaMLP(nn.Module):
    """Gemma MLP with GELUTanh activation (not SiLU like Llama)."""

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size,
            quant_config=quant_config,
        )
        self.act_fn = GeluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)


class GemmaDenseDecoderLayer(nn.Module):
    """Single Gemma decoder layer with dense KV cache support.

    Args:
        config: Object with hidden_size, num_attention_heads,
                num_key_value_heads, head_dim, intermediate_size, rms_norm_eps.
    """

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaDenseAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            hidden_states: (batch, seq, hidden_size)
            new_kv_cache: (key, value) for this layer.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states, cos, sin,
            attention_mask=attention_mask, kv_cache=kv_cache,
            causal=causal,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv
