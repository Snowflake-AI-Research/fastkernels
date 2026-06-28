"""Standalone DeepSeek V3.2 model implementation.

Supports MLA (Multi-head Latent Attention), MoE with grouped routing,
and DSA (DeepSeek Sparse Attention) for V3.2.
Uses YARN-scaled RoPE, FP8 quantization, and tensor parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from ..L1.rms_norm import RMSNorm
from ..L1.yarn_rotary_emb import YarnRotaryEmbedding
from ..L3.deepseek_decoder import DeepSeekDecoderLayer


@dataclass
class DeepSeekV3Config:
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    vocab_size: int = 129280
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # MLA params
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE params
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    # Routing variant.  DeepSeek-V3/V3.2 ship ``scoring_func='sigmoid'`` and
    # ``topk_method='noaux_tc'`` with ``norm_topk_prob=True``.  Older V2
    # checkpoints used ``softmax``.  We default to V2's softmax for backwards
    # compatibility and override from the HF config in ``from_pretrained``.
    scoring_func: str = "softmax"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    hidden_act: str = "silu"

    # DSA params (V3.2 only — None when not a V3.2 model)
    index_topk: Optional[int] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None

    # YARN RoPE params
    rope_parameters: dict = field(default_factory=lambda: {
        'rope_type': 'deepseek_yarn',
        'factor': 40.0,
        'mscale': 1.0,
        'mscale_all_dim': 1.0,
        'attn_factor': 1.0,
        'beta_fast': 32,
        'beta_slow': 1,
        'original_max_position_embeddings': 4096,
    })

    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_name: str) -> "DeepSeekV3Config":
        try:
            hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except ValueError:
            import os as _os
            from transformers import DeepseekV3Config as _HFDSConfig
            import json
            if _os.path.isdir(model_name):
                path = _os.path.join(model_name, "config.json")
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(model_name, "config.json")
            with open(path) as f:
                cfg = json.load(f)
            cfg["model_type"] = "deepseek_v3"
            hf = _HFDSConfig(**cfg)
        rope = getattr(hf, 'rope_scaling', {}) or {}
        rope_params = {
            'rope_type': rope.get('type', rope.get('rope_type', 'deepseek_yarn')),
            'factor': rope.get('factor', 40.0),
            'mscale': rope.get('mscale', 1.0),
            'mscale_all_dim': rope.get('mscale_all_dim', 1.0),
            'attn_factor': rope.get('attn_factor', 1.0),
            'beta_fast': rope.get('beta_fast', 32),
            'beta_slow': rope.get('beta_slow', 1),
            'original_max_position_embeddings': rope.get(
                'original_max_position_embeddings',
                getattr(hf, 'original_max_position_embeddings', 4096)),
        }

        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            moe_intermediate_size=getattr(hf, 'moe_intermediate_size', 2048),
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            rms_norm_eps=getattr(hf, 'rms_norm_eps', 1e-6),
            rope_theta=getattr(hf, 'rope_theta', 10000.0),
            q_lora_rank=getattr(hf, 'q_lora_rank', 1536),
            kv_lora_rank=getattr(hf, 'kv_lora_rank', 512),
            qk_nope_head_dim=getattr(hf, 'qk_nope_head_dim', 128),
            qk_rope_head_dim=getattr(hf, 'qk_rope_head_dim', 64),
            v_head_dim=getattr(hf, 'v_head_dim', 128),
            n_routed_experts=getattr(hf, 'n_routed_experts', 256),
            n_shared_experts=getattr(hf, 'n_shared_experts', 1),
            num_experts_per_tok=getattr(hf, 'num_experts_per_tok', 8),
            n_group=getattr(hf, 'n_group', 8),
            topk_group=getattr(hf, 'topk_group', 4),
            routed_scaling_factor=getattr(hf, 'routed_scaling_factor', 2.5),
            first_k_dense_replace=getattr(hf, 'first_k_dense_replace', 1),
            moe_layer_freq=getattr(hf, 'moe_layer_freq', 1),
            scoring_func=getattr(hf, 'scoring_func', 'softmax'),
            topk_method=getattr(hf, 'topk_method', 'noaux_tc'),
            norm_topk_prob=getattr(hf, 'norm_topk_prob', True),
            hidden_act=getattr(hf, 'hidden_act', 'silu'),
            index_topk=getattr(hf, 'index_topk', None),
            index_n_heads=getattr(hf, 'index_n_heads', None),
            index_head_dim=getattr(hf, 'index_head_dim', None),
            rope_parameters=rope_params,
        )


class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config, quant_config: dict | None = None):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.rotary_emb = YarnRotaryEmbedding(
            head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.rope_parameters.get(
                'original_max_position_embeddings', config.max_position_embeddings),
            rope_theta=config.rope_theta,
            scaling_factor=config.rope_parameters.get('factor', 1.0),
            attn_factor=config.rope_parameters.get('attn_factor', 1.0),
            beta_fast=config.rope_parameters.get('beta_fast', 32),
            beta_slow=config.rope_parameters.get('beta_slow', 1),
            mscale=config.rope_parameters.get('mscale', 1.0),
            mscale_all_dim=config.rope_parameters.get('mscale_all_dim', 0.0),
        )

        is_v32 = hasattr(config, 'index_topk') and config.index_topk is not None

        # Pre-allocate topk_indices_buffer for DSA indexer (shared across layers)
        if is_v32:
            max_batched = getattr(config, 'max_num_batched_tokens', 16384)
            self.topk_indices_buffer = torch.empty(
                max_batched, config.index_topk,
                dtype=torch.int32,
            )
        else:
            self.topk_indices_buffer = None

        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(
                config, layer_idx=i,
                rotary_emb=self.rotary_emb,
                quant_config=quant_config,
                is_v32=is_v32,
                topk_indices_buffer=self.topk_indices_buffer,
            )
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


class DeepSeekV3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "q_a_proj": ("fused_qkv_a_proj", 0),
        "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
    }

    def __init__(self, config: DeepSeekV3Config, quant_config: dict | None = None):
        super().__init__()
        self.config = config
        self.model = DeepSeekV3Model(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

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
