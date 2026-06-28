"""Llama 4 Scout model: MoE with NoPE, QK norm, and temperature tuning.

Uses Llama 4-style architecture with:
  - 16 experts, top-1 sigmoid routing, shared expert on all layers
  - NoPE (no RoPE) on 36/48 layers with attention temperature tuning
  - Weight-less QK RMSNorm after RoPE on RoPE layers
  - Q/K weight permutation (interleaved→contiguous for rotary)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L3.llama4_decoder import Llama4DecoderLayer


@dataclass
class Llama4Config:
    model_type: str = "llama4"
    hidden_size: int = 5120
    intermediate_size: int = 8192
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 202048
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 16.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 1.0
    rope_original_max_position_embeddings: int = 8192
    num_local_experts: int = 16
    num_experts_per_tok: int = 1
    interleave_moe_layer_step: int = 1
    intermediate_size_mlp: int = 16384
    no_rope_layers: list = field(default_factory=list)
    attention_chunk_size: int | None = None
    use_qk_norm: bool = True
    attn_temperature_tuning: bool = True
    attn_scale: float = 0.1
    floor_scale: float = 8192.0
    tie_word_embeddings: bool = False
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Llama4Config":
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Llama4 uses nested text_config
        text = getattr(hf, "text_config", hf)
        rope = getattr(text, "rope_scaling", None) or {}
        rope_theta = rope.get("rope_theta") or getattr(text, "rope_theta", 500000.0)
        no_rope_layers = getattr(text, "no_rope_layers", [])
        return cls(
            hidden_size=text.hidden_size,
            intermediate_size=text.intermediate_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            head_dim=getattr(text, "head_dim", text.hidden_size // text.num_attention_heads),
            vocab_size=text.vocab_size,
            max_position_embeddings=min(
                getattr(text, "max_position_embeddings", 131072), 131072,
            ),
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=rope.get("factor", 1.0),
            rope_low_freq_factor=rope.get("low_freq_factor", 1.0),
            rope_high_freq_factor=rope.get("high_freq_factor", 1.0),
            rope_original_max_position_embeddings=rope.get(
                "original_max_position_embeddings", 8192,
            ),
            num_local_experts=text.num_local_experts,
            num_experts_per_tok=text.num_experts_per_tok,
            interleave_moe_layer_step=getattr(text, "interleave_moe_layer_step", 1),
            intermediate_size_mlp=getattr(text, "intermediate_size_mlp", 16384),
            no_rope_layers=no_rope_layers if isinstance(no_rope_layers, list) else list(no_rope_layers),
            attention_chunk_size=getattr(text, "attention_chunk_size", None),
            use_qk_norm=getattr(text, "use_qk_norm", True),
            attn_temperature_tuning=getattr(text, "attn_temperature_tuning", True),
            attn_scale=getattr(text, "attn_scale", 0.1),
            floor_scale=getattr(text, "floor_scale", 8192.0),
            tie_word_embeddings=getattr(text, "tie_word_embeddings", False),
        )


class Llama4Model(nn.Module):
    def __init__(self, config: Llama4Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_low_freq_factor=config.rope_low_freq_factor,
            rope_high_freq_factor=config.rope_high_freq_factor,
            rope_original_max_position_embeddings=config.rope_original_max_position_embeddings,
        )
        self.layers = nn.ModuleList([
            Llama4DecoderLayer(config, layer_idx=i, rotary_emb=self.rotary_emb)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Llama4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Llama4Config):
        super().__init__()
        self.config = config
        self.model = Llama4Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits):
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits):
        result = self.lm_head.gather_greedy(partial_logits.float())
        return result
