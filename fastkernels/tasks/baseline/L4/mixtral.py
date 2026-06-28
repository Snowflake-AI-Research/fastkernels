"""Standalone Mixtral-8x7B model implementation.

Adds Mixture-of-Experts (MoE) with top-k gating on top of the shared
attention and normalization layers. Uses standard RoPE (no frequency scaling).
Supports tensor parallelism via shared TP layers.

The MoE layer uses a fused Triton grouped-GEMM kernel for high throughput.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L3.mixtral_decoder import MixtralDecoderLayer


@dataclass
class MixtralConfig:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 32000
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MixtralConfig":
        hf = AutoConfig.from_pretrained(model_name)
        head_dim = getattr(hf, "head_dim", None)
        if head_dim is None:
            head_dim = hf.hidden_size // hf.num_attention_heads
        rope = getattr(hf, "rope_parameters", None) or {}
        rope_theta = getattr(hf, "rope_theta", None) or rope.get("rope_theta", 1000000.0)
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=rope_theta,
            num_local_experts=hf.num_local_experts,
            num_experts_per_tok=hf.num_experts_per_tok,
        )


class MixtralModel(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, rotary_emb=self.rotary_emb)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.model = MixtralModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits
