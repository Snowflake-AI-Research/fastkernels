"""Standalone Qwen3-Next model implementation.

Matches Qwen/Qwen3-Next-80B-A3B-Instruct architecture:
    - 48 layers, 3:1 pattern (36 GDN linear attention + 12 full attention)
    - GDN: 16 key heads, 32 value heads, key_dim=128, value_dim=128
    - Full attention: 16 query heads, 2 KV heads, head_dim=256 (GQA)
    - Per-head QK-RMSNorm, partial RoPE (25% of head_dim = 64 dims)
    - Output gating on full attention: o * sigmoid(gate)
    - MoE: 512 routed experts (top-10, softmax), shared expert with sigmoid gate
    - moe_intermediate_size=512, shared_expert_intermediate_size=512
    - hidden_size=2048, vocab_size=151936
    - GemmaRMSNorm (weight + 1 convention)
    - rope_theta=10,000,000

Weight names match HuggingFace checkpoint convention:
    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.linear_attn.in_proj_qkvz.weight
    model.layers.{i}.linear_attn.in_proj_ba.weight
    model.layers.{i}.linear_attn.conv1d.weight
    model.layers.{i}.linear_attn.A_log
    model.layers.{i}.linear_attn.dt_bias
    model.layers.{i}.linear_attn.norm.weight
    model.layers.{i}.linear_attn.out_proj.weight
    model.layers.{i}.self_attn.q_proj.weight
    model.layers.{i}.self_attn.k_proj.weight
    model.layers.{i}.self_attn.v_proj.weight
    model.layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.self_attn.q_norm.weight
    model.layers.{i}.self_attn.k_norm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.gate.weight
    model.layers.{i}.mlp.shared_expert_gate.weight
    model.layers.{i}.mlp.shared_expert.gate_proj.weight
    model.layers.{i}.mlp.shared_expert.up_proj.weight
    model.layers.{i}.mlp.shared_expert.down_proj.weight
    model.layers.{i}.mlp.experts.{j}.gate_proj.weight
    model.layers.{i}.mlp.experts.{j}.up_proj.weight
    model.layers.{i}.mlp.experts.{j}.down_proj.weight
    model.norm.weight
    lm_head.weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.qwen3_next_decoder import Qwen3NextDecoderLayer


def _default_layer_types() -> list[str]:
    """3:1 pattern: linear_attention for (i+1)%4 != 0, else full_attention."""
    return [
        "linear_attention" if bool((i + 1) % 4) else "full_attention"
        for i in range(48)
    ]


@dataclass
class Qwen3NextConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 5632
    vocab_size: int = 151936
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 0.25
    hidden_act: str = "silu"
    attention_bias: bool | None = None
    # Linear attention (GDN)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)
    # Layer types
    layer_types: list[str] = field(default_factory=_default_layer_types)
    # Misc
    tie_word_embeddings: bool = False
    model_type: str = "qwen3_next"
    dtype: torch.dtype = torch.bfloat16

    def is_linear_attn_layer(self, layer_idx: int) -> bool:
        """``True`` if ``layer_idx`` is a Gated-DeltaNet (linear) layer."""
        return self.layer_types[layer_idx] == "linear_attention"

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "Qwen3NextConfig":
        try:
            hf = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            return cls._from_hf(hf)
        except Exception:
            path = Path(model_name_or_path) / "config.json"
            with path.open() as f:
                data = json.load(f)
            return cls._from_dict(data)

    @classmethod
    def _from_hf(cls, hf) -> "Qwen3NextConfig":
        rope_params = getattr(hf, "rope_parameters", None) or {}
        rope_theta = rope_params.get(
            "rope_theta", getattr(hf, "rope_theta", 10_000_000.0)
        )
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor",
            getattr(hf, "partial_rotary_factor", 0.25),
        )
        layer_types = getattr(hf, "layer_types", None)
        if layer_types is None:
            n = getattr(hf, "num_hidden_layers", 48)
            layer_types = [
                "linear_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(n)
            ]
        return cls(
            hidden_size=hf.hidden_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, "num_key_value_heads", 2),
            head_dim=getattr(hf, "head_dim", 256),
            intermediate_size=getattr(hf, "intermediate_size", 5632),
            vocab_size=hf.vocab_size,
            max_position_embeddings=getattr(hf, "max_position_embeddings", 262144),
            rms_norm_eps=getattr(hf, "rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            rope_scaling=getattr(hf, "rope_scaling", None),
            partial_rotary_factor=partial_rotary_factor,
            hidden_act=getattr(hf, "hidden_act", "silu"),
            attention_bias=getattr(hf, "attention_bias", None),
            linear_conv_kernel_dim=getattr(hf, "linear_conv_kernel_dim", 4),
            linear_key_head_dim=getattr(hf, "linear_key_head_dim", 128),
            linear_value_head_dim=getattr(hf, "linear_value_head_dim", 128),
            linear_num_key_heads=getattr(hf, "linear_num_key_heads", 16),
            linear_num_value_heads=getattr(hf, "linear_num_value_heads", 32),
            num_experts=getattr(hf, "num_experts", 512),
            num_experts_per_tok=getattr(hf, "num_experts_per_tok", 10),
            moe_intermediate_size=getattr(hf, "moe_intermediate_size", 512),
            shared_expert_intermediate_size=getattr(
                hf, "shared_expert_intermediate_size", 512
            ),
            norm_topk_prob=getattr(hf, "norm_topk_prob", True),
            decoder_sparse_step=getattr(hf, "decoder_sparse_step", 1),
            mlp_only_layers=getattr(hf, "mlp_only_layers", []),
            layer_types=layer_types,
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
        )

    @classmethod
    def _from_dict(cls, data: dict) -> "Qwen3NextConfig":
        rope_params = data.get("rope_parameters", {}) or {}
        rope_scaling = data.get("rope_scaling", {}) or {}
        merged = {**rope_params, **rope_scaling}
        rope_theta = merged.get(
            "rope_theta", data.get("rope_theta", 10_000_000.0)
        )
        partial_rotary_factor = merged.get(
            "partial_rotary_factor",
            data.get("partial_rotary_factor", 0.25),
        )
        n = data.get("num_hidden_layers", 48)
        layer_types = data.get("layer_types", None)
        if layer_types is None:
            layer_types = [
                "linear_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(n)
            ]
        return cls(
            hidden_size=data.get("hidden_size", 2048),
            num_hidden_layers=n,
            num_attention_heads=data.get("num_attention_heads", 16),
            num_key_value_heads=data.get("num_key_value_heads", 2),
            head_dim=data.get("head_dim", 256),
            intermediate_size=data.get("intermediate_size", 5632),
            vocab_size=data.get("vocab_size", 151936),
            max_position_embeddings=data.get("max_position_embeddings", 262144),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            rope_scaling=data.get("rope_scaling", None),
            partial_rotary_factor=partial_rotary_factor,
            hidden_act=data.get("hidden_act", "silu"),
            attention_bias=data.get("attention_bias", None),
            linear_conv_kernel_dim=data.get("linear_conv_kernel_dim", 4),
            linear_key_head_dim=data.get("linear_key_head_dim", 128),
            linear_value_head_dim=data.get("linear_value_head_dim", 128),
            linear_num_key_heads=data.get("linear_num_key_heads", 16),
            linear_num_value_heads=data.get("linear_num_value_heads", 32),
            num_experts=data.get("num_experts", 512),
            num_experts_per_tok=data.get("num_experts_per_tok", 10),
            moe_intermediate_size=data.get("moe_intermediate_size", 512),
            shared_expert_intermediate_size=data.get(
                "shared_expert_intermediate_size", 512
            ),
            norm_topk_prob=data.get("norm_topk_prob", True),
            decoder_sparse_step=data.get("decoder_sparse_step", 1),
            mlp_only_layers=data.get("mlp_only_layers", []),
            layer_types=layer_types,
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )


class Qwen3NextModel(nn.Module):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3NextDecoderLayer(config, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE: partial rotary (only rotary_dim = partial_rotary_factor * head_dim)
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        self.rotary_emb = RotaryEmbedding(
            rotary_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def forward(self, input_ids, positions=None, state_manager=None):
        if input_ids.dim() > 1:
            input_ids = input_ids.reshape(-1)
        if positions is not None and positions.dim() > 1:
            positions = positions.reshape(-1)
        hidden_states = self.embed_tokens(input_ids)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual,
                positions=positions,
                rotary_emb=self.rotary_emb,
                state_manager=state_manager,
            )

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3NextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3NextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions=None, state_manager=None):
        return self.model(input_ids, positions, state_manager=state_manager)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, hidden_states):
        return self.compute_logits(hidden_states)
