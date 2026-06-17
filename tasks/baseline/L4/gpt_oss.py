"""Standalone GPT-OSS model implementation.

Matches openai/gpt-oss-20b architecture:
    - 24 layers, alternating sliding_attention / full_attention
    - hidden_size=2880, 64 query heads, 8 KV heads, head_dim=64 (GQA)
    - 32 experts, top-4 softmax routing, intermediate_size=2880
    - SwiGLU with clamp limit (swiglu_limit=7.0)
    - YaRN RoPE (theta=150000, factor=32, original_max=4096)
    - Sliding window=128 on even layers
    - Attention sinks (learnable per-head biases)
    - Bias on QKV, O, router, and expert projections
    - MXFP4 quantized expert weights (native MXFP4 Triton inference)
    - vocab_size=201088

Weight names match HuggingFace checkpoint convention:
    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.q_proj.weight/bias
    model.layers.{i}.self_attn.k_proj.weight/bias
    model.layers.{i}.self_attn.v_proj.weight/bias
    model.layers.{i}.self_attn.o_proj.weight/bias
    model.layers.{i}.self_attn.sinks
    model.layers.{i}.mlp.router.weight/bias
    model.layers.{i}.mlp.experts.gate_up_proj_blocks/scales/bias
    model.layers.{i}.mlp.experts.down_proj_blocks/scales/bias
    model.norm.weight
    lm_head.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.rms_norm import RMSNorm
from ..L1.yarn_rotary_emb import YaRNRotaryEmbedding
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.gpt_oss_decoder import GptOssDecoderLayer


@dataclass
class GptOssConfig:
    hidden_size: int = 2880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2880
    vocab_size: int = 201088
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 150000.0
    # YaRN
    rope_scaling_factor: float = 32.0
    rope_original_max_position_embeddings: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_truncate: bool = False
    # MoE
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    swiglu_limit: float = 7.0
    # Sliding window
    sliding_window: int = 128
    # Misc
    tie_word_embeddings: bool = False
    model_type: str = "gpt_oss"
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "GptOssConfig":
        try:
            hf = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            return cls._from_hf(hf)
        except Exception:
            path = Path(model_name_or_path) / "config.json"
            with path.open() as f:
                data = json.load(f)
            return cls._from_dict(data)

    @classmethod
    def _from_hf(cls, hf) -> "GptOssConfig":
        rope_scaling = getattr(hf, "rope_scaling", None) or {}
        rope_params = getattr(hf, "rope_parameters", None) or {}
        # Merge both sources
        rs = {**rope_params, **rope_scaling}
        return cls(
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            intermediate_size=hf.intermediate_size,
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=getattr(hf, "rms_norm_eps", 1e-5),
            rope_theta=rs.get("rope_theta", getattr(hf, "rope_theta", 150000.0)),
            rope_scaling_factor=rs.get("factor", 32.0),
            rope_original_max_position_embeddings=rs.get(
                "original_max_position_embeddings", 4096,
            ),
            rope_beta_fast=rs.get("beta_fast", 32.0),
            rope_beta_slow=rs.get("beta_slow", 1.0),
            rope_truncate=rs.get("truncate", False),
            num_local_experts=getattr(hf, "num_local_experts", 32),
            num_experts_per_tok=getattr(hf, "num_experts_per_tok", 4),
            swiglu_limit=getattr(hf, "swiglu_limit", 7.0),
            sliding_window=getattr(hf, "sliding_window", 128),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
        )

    @classmethod
    def _from_dict(cls, data: dict) -> "GptOssConfig":
        rs = data.get("rope_scaling", {}) or {}
        rp = data.get("rope_parameters", {}) or {}
        merged = {**rp, **rs}
        return cls(
            hidden_size=data.get("hidden_size", 2880),
            num_hidden_layers=data.get("num_hidden_layers", 24),
            num_attention_heads=data.get("num_attention_heads", 64),
            num_key_value_heads=data.get("num_key_value_heads", 8),
            head_dim=data.get("head_dim", 64),
            intermediate_size=data.get("intermediate_size", 2880),
            vocab_size=data.get("vocab_size", 201088),
            max_position_embeddings=data.get("max_position_embeddings", 131072),
            rms_norm_eps=data.get("rms_norm_eps", 1e-5),
            rope_theta=merged.get("rope_theta", data.get("rope_theta", 150000.0)),
            rope_scaling_factor=merged.get("factor", 32.0),
            rope_original_max_position_embeddings=merged.get(
                "original_max_position_embeddings", 4096,
            ),
            rope_beta_fast=merged.get("beta_fast", 32.0),
            rope_beta_slow=merged.get("beta_slow", 1.0),
            rope_truncate=merged.get("truncate", False),
            num_local_experts=data.get("num_local_experts", 32),
            num_experts_per_tok=data.get("num_experts_per_tok", 4),
            swiglu_limit=data.get("swiglu_limit", 7.0),
            sliding_window=data.get("sliding_window", 128),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = YaRNRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
            original_max_position_embeddings=config.rope_original_max_position_embeddings,
            beta_fast=config.rope_beta_fast,
            beta_slow=config.rope_beta_slow,
            truncate=config.rope_truncate,
        )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, self.rotary_emb,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GptOssForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits
