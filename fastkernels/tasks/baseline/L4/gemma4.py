"""Gemma4 text-only causal LM.

Implements the ``google/gemma-4-26B-A4B-it`` language stack from the nested
``Gemma4ForConditionalGeneration`` checkpoint and skips multimodal weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import Gemma4ProportionalRotaryEmbedding, RotaryEmbedding
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.gemma4_decoder import Gemma4DecoderLayer


def _read_config(model_name: str) -> dict:
    if os.path.isdir(model_name):
        path = os.path.join(model_name, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_name, "config.json")
    with open(path) as f:
        return json.load(f)


@dataclass
class Gemma4Config:
    model_type: str = "gemma4"
    hidden_size: int = 2816
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    vocab_size: int = 262144
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_k_eq_v: bool = True
    hidden_activation: str = "gelu_pytorch_tanh"
    sliding_window: int = 1024
    num_experts: int = 128
    top_k_experts: int = 8
    enable_moe_block: bool = True
    tie_word_embeddings: bool = True
    final_logit_softcapping: float | None = 30.0
    rope_parameters: dict = field(default_factory=lambda: {
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "full_attention": {
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
        },
    })
    layer_types: list[str] = field(default_factory=list)
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Gemma4Config":
        raw = _read_config(model_name)
        text = raw.get("text_config", raw)
        return cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            moe_intermediate_size=text["moe_intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            num_global_key_value_heads=text.get("num_global_key_value_heads", 2),
            head_dim=text["head_dim"],
            global_head_dim=text.get("global_head_dim", text["head_dim"]),
            vocab_size=text["vocab_size"],
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=text["rms_norm_eps"],
            attention_bias=text.get("attention_bias", False),
            attention_k_eq_v=text.get("attention_k_eq_v", False),
            hidden_activation=text.get("hidden_activation", "gelu_pytorch_tanh"),
            sliding_window=text.get("sliding_window", 1024),
            num_experts=text["num_experts"],
            top_k_experts=text["top_k_experts"],
            enable_moe_block=text.get("enable_moe_block", False),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            final_logit_softcapping=text.get("final_logit_softcapping"),
            rope_parameters=text["rope_parameters"],
            layer_types=list(text["layer_types"]),
        )


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
        )
        self.embed_scale = math.sqrt(config.hidden_size)

        self.rotary_embeddings = nn.ModuleDict()
        rotary_dims: dict[str, int] = {}
        for layer_type in set(config.layer_types):
            is_sliding = layer_type == "sliding_attention"
            head_dim = config.head_dim if is_sliding else config.global_head_dim
            rope = config.rope_parameters[layer_type]
            rotary_dim = int(head_dim * rope.get("partial_rotary_factor", 1.0))
            if rope.get("rope_type") == "proportional":
                rotary_dims[layer_type] = head_dim
                self.rotary_embeddings[layer_type] = Gemma4ProportionalRotaryEmbedding(
                    head_dim,
                    rotary_dim,
                    config.max_position_embeddings,
                    rope.get("rope_theta", 10000.0),
                )
            else:
                rotary_dims[layer_type] = rotary_dim
                self.rotary_embeddings[layer_type] = RotaryEmbedding(
                    rotary_dim,
                    config.max_position_embeddings,
                    rope.get("rope_theta", 10000.0),
                )

        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(
                config,
                i,
                self.rotary_embeddings[config.layer_types[i]],
                rotary_dims[config.layer_types[i]],
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        hidden_states = hidden_states * self.embed_scale
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Gemma4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.embedding_op.emb.weight = (
                self.model.embed_tokens.embedding_op.emb.weight
            )

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def forward_with_lm_proj(self, input_ids, positions):
        return self.lm_head.project(self.model(input_ids, positions))

    def _softcap(self, logits):
        cap = self.config.final_logit_softcapping
        if logits is not None and cap is not None:
            logits = cap * torch.tanh(logits.float() / cap)
        elif logits is not None:
            logits = logits.float()
        return logits

    def compute_logits(self, hidden_states):
        return self._softcap(self.lm_head(hidden_states))

    def compute_logits_no_softcap(self, hidden_states):
        return self.lm_head(hidden_states)

    def compute_logits_decode(self, partial_logits):
        return self._softcap(self.lm_head.gather_logits(partial_logits))

    def greedy_sample_decode(self, partial_logits):
        return self.lm_head.gather_greedy(partial_logits.float())
