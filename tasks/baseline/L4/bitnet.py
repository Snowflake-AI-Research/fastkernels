"""Standalone BitNet b1.58 model implementation.

Targets ``microsoft/bitnet-b1.58-2B-4T`` with the "Native 1.58-bit weights
and 8-bit activations" (W1.58A8) format:

    * GQA attention (20 query heads / 5 KV heads, head_dim=128)
    * NeoX-style RoPE (theta=500000, max_position=4096)
    * Squared-ReLU gated MLP (no SwiGLU)
    * Per-token int8 activation quantization in every projection
      (see ``L1/bitnet_linear.py``)
    * Per-tensor bf16 ``weight_scale`` recovered from packed uint8 weights
    * Attention and FFN sub-RMSNorms
    * Tied embeddings (lm_head shares weight with embed_tokens)

Weight names follow the HuggingFace checkpoint convention so that the shared
weight loader can populate them directly:

    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight (+ .weight_scale)
    model.layers.{i}.self_attn.attn_sub_norm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight  (+ .weight_scale)
    model.layers.{i}.mlp.ffn_sub_norm.weight
    model.norm.weight
    lm_head.weight  (tied with model.embed_tokens.weight)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.bitnet_rms_norm import BitNetRMSNorm as RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L3.bitnet_decoder import BitNetDecoderLayer


@dataclass
class BitNetConfig:
    model_type: str = "bitnet"
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    head_dim: int = 128
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "BitNetConfig":
        # BitNet ships an ``auto_map`` pointing at remote .py files that
        # are not actually present in the repo, so loading with
        # ``trust_remote_code=True`` fails.  ``model_type: "bitnet"`` is
        # registered natively in transformers, so loading without
        # ``trust_remote_code`` works on recent transformers releases.
        try:
            hf = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        except (KeyError, ValueError):
            hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        head_dim = getattr(hf, "head_dim", None)
        if head_dim is None:
            head_dim = hf.hidden_size // hf.num_attention_heads
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
            rope_theta=hf.rope_theta,
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", True),
        )


class BitNetModel(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
            is_neox_style=False,
        )
        self.layers = nn.ModuleList([
            BitNetDecoderLayer(config, rotary_emb=self.rotary_emb)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class BitNetForCausalLM(nn.Module):
    # BitNet's checkpoint stores attention projections separately
    # (q_proj, k_proj, v_proj) and MLP gate / up separately, but kb_nano
    # consolidates each group into a single ``BitLinearMerged`` for SOTA
    # parity with vllm_repo/BitNet's ``wqkv`` and ``w13`` and to halve
    # GEMM kernel launch overhead.  ``packed_modules_mapping`` tells the
    # shared weight loader how to redirect the on-disk tensor names into
    # the fused params, with the ``shard_id`` carrying the sub-projection
    # tag.  Substring matching in the loader handles both ``.weight`` and
    # ``.weight_scale`` automatically.
    packed_modules_mapping: dict = {
        "q_proj":    ("qkv_proj", "q"),
        "k_proj":    ("qkv_proj", "k"),
        "v_proj":    ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":   ("gate_up_proj", 1),
    }

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.model = BitNetModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # ParallelLMHead inherits from VocabParallelEmbedding, so the
            # underlying nn.Embedding parameter must be shared.
            self.lm_head.embedding_op.emb.weight = (
                self.model.embed_tokens.embedding_op.emb.weight
            )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def forward_with_lm_proj(self, input_ids, positions):
        hidden_states = self.model(input_ids, positions)
        return self.lm_head.project(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits: torch.Tensor):
        return self.lm_head.gather_greedy(partial_logits.float())
