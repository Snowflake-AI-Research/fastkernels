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
    # Indexer RoPE layout: DeepSeek-V3.2 uses NeoX (interleave=False); GLM-5.2
    # (``glm_moe_dsa``) sets ``indexer_rope_interleave=True`` (interleaved).
    # Consumed by the DSA indexer as ``is_neox_style = not indexer_rope_interleave``.
    indexer_rope_interleave: bool = False
    # DSA index-topk sharing: most layers SKIP computing their own top-k index
    # and REUSE the last compute layer's (via the shared ``topk_indices_buffer``).
    # Per-layer skip is derived from these exactly as vLLM's DeepseekV2MLAAttention
    # (``deepseek_v2.py:1003-1018``). Defaults (freq=1) => every layer computes,
    # so DeepSeek-V3.2 is unchanged; GLM-5.2 ships freq=4, offset=3.
    index_topk_freq: int = 1
    index_topk_pattern: Optional[list] = None
    index_skip_topk_offset: int = 2
    # Sizes the shared DSA topk_indices_buffer. vLLM uses
    # scheduler_config.max_num_batched_tokens; this is not an HF-config field, so
    # the engine threads its real value in via load_model before construction.
    max_num_batched_tokens: int = 16384
    # MLA paged-KV cache dtype, which also selects the attention backend:
    # ``"auto"`` (BF16 cache -> FlashMLA sparse), ``"fp8_ds_mla"`` (DeepSeek's
    # 656-byte block-scaled cache), ``"fp8_e4m3"`` (plain per-tensor fp8 ->
    # vLLM's FLASHINFER_MLA_SPARSE). ``None`` defers to
    # ``FASTKERNELS_KV_CACHE_DTYPE`` (default ``"auto"``).
    kv_cache_dtype: Optional[str] = None

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
        # DeepSeek-V3.2 carries its YARN settings under ``rope_scaling`` with a
        # top-level ``rope_theta``. GLM-5.2 (``glm_moe_dsa``) instead uses the
        # newer ``rope_parameters`` block with ``rope_type: "default"`` (plain
        # RoPE, no YARN) and ``rope_theta`` nested inside it. Parse both, and
        # gate the plain-RoPE branch so DeepSeek stays byte-identical.
        rope = getattr(hf, 'rope_scaling', None) or {}
        rope_params_hf = getattr(hf, 'rope_parameters', None) or {}
        rope_type = (
            rope.get('type') or rope.get('rope_type')
            or rope_params_hf.get('rope_type') or 'deepseek_yarn'
        )
        # theta: the ``rope_parameters`` block is authoritative when present
        # (GLM-5.2 nests ``rope_theta: 8e6`` there and has no top-level value);
        # otherwise use the top-level ``rope_theta`` (DeepSeek-V3.2 -> 10000).
        # NB: the AutoConfig-fallback path builds an HF ``DeepseekV3Config`` which
        # injects a spurious default ``rope_theta=10000``, so we must not let a
        # top-level value shadow the nested GLM one.
        rope_theta = rope_params_hf.get('rope_theta')
        if rope_theta is None:
            rope_theta = rope.get('rope_theta')  # migrated into rope_scaling
        if rope_theta is None:
            rope_theta = getattr(hf, 'rope_theta', 10000.0)

        if rope_type in ('default', 'plain', 'linear'):
            # Plain RoPE (GLM-5.2). Gate on ``rope_type`` -- not on ``rope`` being
            # empty -- because some transformers versions migrate the new
            # ``rope_parameters`` block into ``rope_scaling``, so ``rope`` is
            # non-empty even for GLM. DeepSeek-V3.2 uses ``rope_type='yarn'`` /
            # ``'deepseek_yarn'`` (never in this set), so it keeps the else path.
            # factor=1.0 makes YarnRotaryEmbedding degrade
            # to standard RoPE (softmax_mscale=1.0, inv_freq == 1/pos_freqs), and
            # the cache must span the full context (original_max = max_position).
            rope_params = {
                'rope_type': rope_type,
                'factor': 1.0,
                'mscale': 1.0,
                'mscale_all_dim': 0.0,
                'attn_factor': 1.0,
                'beta_fast': 32,
                'beta_slow': 1,
                'original_max_position_embeddings': hf.max_position_embeddings,
            }
        else:
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
            rope_theta=rope_theta,
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
            indexer_rope_interleave=getattr(hf, 'indexer_rope_interleave', False),
            index_topk_freq=getattr(hf, 'index_topk_freq', 1),
            index_topk_pattern=getattr(hf, 'index_topk_pattern', None),
            index_skip_topk_offset=getattr(hf, 'index_skip_topk_offset', 2),
            rope_parameters=rope_params,
        )


class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config, quant_config: dict | None = None):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # GLM-5.2 uses plain RoPE (rope_type "default"); vLLM routes that to the
        # base RotaryEmbedding (no FlashInfer, bf16 cos/sin cache). Flag it so
        # YarnRotaryEmbedding matches, while DeepSeek-V3.2 YARN keeps FlashInfer.
        is_plain_rope = config.rope_parameters.get('rope_type') in (
            'default', 'plain', 'linear')
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
            is_plain=is_plain_rope,
            # Plain-rope (GLM-5.2) stores the cos/sin cache in the compute dtype
            # once, matching vLLM's base RotaryEmbedding (no per-forward re-cast).
            cache_dtype=config.dtype,
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

        def _layer_skip_topk(layer_id: int) -> bool:
            """DSA index-topk sharing: which backbone layers REUSE a prior layer's
            top-k index instead of recomputing it. Matches vLLM
            ``deepseek_v2.py:1011-1017``. Non-v32 (or freq==1) => always compute."""
            if not is_v32:
                return False
            pat = getattr(config, 'index_topk_pattern', None)
            if pat is None:
                freq = getattr(config, 'index_topk_freq', 1)
                off = getattr(config, 'index_skip_topk_offset', 2)
                return (max(layer_id - off + 1, 0) % freq) != 0
            if 0 <= layer_id < len(pat):
                return pat[layer_id] == "S"
            return False

        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(
                config, layer_idx=i,
                rotary_emb=self.rotary_emb,
                quant_config=quant_config,
                is_v32=is_v32,
                skip_topk=_layer_skip_topk(i),
                topk_indices_buffer=self.topk_indices_buffer,
                kv_cache_dtype=getattr(config, "kv_cache_dtype", None),
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
