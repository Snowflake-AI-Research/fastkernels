"""DeepSeek MLA attention (model-level).

Consolidates projections (fused_qkv_a_proj, q_a_layernorm, q_b_proj,
kv_a_layernorm, kv_b_proj, o_proj) and dispatches to MLAAttention
for cache storage and kernel execution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from .parallel_linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
)
from .mla_attention_impl import MLAAttention
from .sparse_attn_indexer import SparseAttnIndexer
from ..L1.rms_norm import RMSNorm
from ..L1.yarn_rotary_emb import YarnRotaryEmbedding, yarn_get_mscale


class DeepSeekMLAAttention(nn.Module):
    """DeepSeek Multi-head Latent Attention with optional DSA indexer.

    Forward: fused_qkv_a_proj -> norms -> q_b_proj/kv_b_proj -> RoPE
             -> [Indexer] -> MLA attention -> o_proj
    """

    def __init__(self, config, rotary_emb: nn.Module,
                 quant_config: dict | None = None,
                 is_v32: bool = False,
                 topk_indices_buffer: torch.Tensor | None = None):
        super().__init__()
        tp = _tp_size()
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // tp

        # Scaling
        self.scaling = self.qk_head_dim ** -0.5

        # Apply YARN mscale to scaling
        if hasattr(config, 'rope_parameters'):
            rp = config.rope_parameters
            if rp.get('rope_type') in ('deepseek_yarn', 'yarn'):
                mscale_all_dim = rp.get('mscale_all_dim', 0)
                scaling_factor = rp.get('factor', 1.0)
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale

        self.rotary_emb = rotary_emb
        self.is_v32 = is_v32

        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            disable_tp=True,
        )

        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            quant_config=quant_config,
        )

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            quant_config=quant_config,
        )

        # MLA attention core
        self.attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            is_sparse=self.is_v32,
        )
        # Share the kv_b_proj module so ``MLAAttention.forward_impl`` (and
        # the ``kb_nano::unified_mla_attention`` custom op) can project
        # kv_c_normed -> kv without routing a non-tensor arg through the
        # op schema. Bypass ``nn.Module.__setattr__`` to avoid double
        # registration as a submodule.
        object.__setattr__(self.attn, "_kv_b_proj", self.kv_b_proj)

        # DSA Indexer (V3.2 only)
        if self.is_v32:
            _rp = getattr(config, "rope_parameters", None) or {}
            self.indexer = SparseAttnIndexer(
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                n_head=config.index_n_heads,
                head_dim=config.index_head_dim,
                rope_dim=self.qk_rope_head_dim,
                topk_tokens=config.index_topk,
                quant_config=quant_config,
                topk_indices_buffer=topk_indices_buffer,
            )
            # Indexer RoPE is built from the *same* source as the main attention
            # RoPE (matches ``vllm/model_executor/models/deepseek_v2.py:944-949``
            # which calls ``get_rope(qk_rope_head_dim, max_position_embeddings,
            # rope_parameters=config.rope_parameters, is_neox_style=...)``).
            # The only divergence from main rope is ``is_neox_style``, which is
            # ``not indexer_rope_interleave``.
            indexer_interleave = getattr(config, "indexer_rope_interleave", False)
            self.indexer_rope_emb = YarnRotaryEmbedding(
                head_dim=self.qk_rope_head_dim,
                max_position_embeddings=_rp.get(
                    'original_max_position_embeddings',
                    config.max_position_embeddings),
                rope_theta=getattr(config, "rope_theta", 10000.0),
                scaling_factor=_rp.get("factor", 1.0),
                attn_factor=_rp.get("attn_factor", 1.0),
                beta_fast=_rp.get("beta_fast", 32),
                beta_slow=_rp.get("beta_slow", 1),
                mscale=_rp.get("mscale", 1.0),
                mscale_all_dim=_rp.get("mscale_all_dim", 0.0),
                is_neox_style=not indexer_interleave,
            )
            # Share the rope_emb module with the indexer so its custom op
            # doesn't need a non-tensor argument (same reasoning as
            # ``self.attn._kv_b_proj`` above).
            object.__setattr__(self.indexer, "_rope_emb", self.indexer_rope_emb)
        else:
            self.indexer = None
            self.indexer_rope_emb = None

    def compute_absorbed_weights(self):
        """Compute W_UV and W_UK_T from kv_b_proj for absorbed MLA decode.

        Must be called after weight loading but BEFORE FP8 postprocessing
        (transform_sf_into_required_layout). For FP8 weights, we dequantize
        using the original block scales to recover accurate BF16 values,
        matching vLLM's get_and_maybe_dequant_weights.

        Produces:
        - W_UV: [N, L, V] for v_up_proj (matches vllm)
        - W_UK_T: [N, P, L] for decode query absorption (matches vllm)
        """
        weight = self.kv_b_proj.weight.data
        if hasattr(self.kv_b_proj, 'use_fp8') and self.kv_b_proj.use_fp8:
            scale = self.kv_b_proj.weight_scale_inv.data
            weight = self._dequant_fp8_block(weight, scale)
        else:
            weight = weight.to(torch.bfloat16)
        weight = weight.T  # [L, N*(P+V)]
        L = self.kv_lora_rank
        N = self.num_local_heads
        P = self.qk_nope_head_dim
        V = self.v_head_dim
        weight = weight.view(L, N, P + V)
        W_UK = weight[:, :, :P]  # [L, N, P]
        W_UV = weight[:, :, P:]  # [L, N, V]
        # W_UV: (L, N, V) -> (N, L, V) — matches vllm
        self.attn.W_UV = W_UV.permute(1, 0, 2).contiguous()
        # W_UK_T: (L, N, P) -> (N, P, L) — matches vllm
        self.attn.W_UK_T = W_UK.permute(1, 2, 0).contiguous()

    @staticmethod
    def _dequant_fp8_block(w_fp8: torch.Tensor, scale_inv: torch.Tensor,
                           block_size: int = 128) -> torch.Tensor:
        """Dequantize block-scaled FP8 weight [N, K] to BF16 using per-block scales."""
        import math
        N, K = w_fp8.shape
        sn = math.ceil(N / block_size)
        sk = math.ceil(K / block_size)
        scale = scale_inv[:sn, :sk]
        scale_expanded = scale.repeat_interleave(block_size, dim=0)[:N] \
                              .repeat_interleave(block_size, dim=1)[:, :K]
        return (w_fp8.float() * scale_expanded).to(torch.bfloat16)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        N = hidden_states.shape[0]

        # Fused Q + KV_a projection
        qkv_lora = self.fused_qkv_a_proj(hidden_states)
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)

        # Q path
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)  # [N, num_local_heads * qk_head_dim]
        q = q.view(N, self.num_local_heads, self.qk_head_dim)

        # KV path
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)  # [N, 1, qk_rope_head_dim]

        # RoPE on q_pe and k_pe
        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe)

        # DSA Indexer (V3.2) — rope_emb is wired to self.indexer._rope_emb
        # so the forward takes only tensor args (custom-op safe).
        topk_indices = None
        if self.indexer is not None and self.is_v32:
            topk_indices = self.indexer(hidden_states, q_c, positions)

        # MLA attention — kv_b_proj is wired to self.attn._kv_b_proj so
        # the forward takes only tensor args (custom-op safe).
        attn_output = self.attn(
            q, kv_c_normed, k_pe,
            topk_indices=topk_indices,
            output_shape=(N, self.num_local_heads * self.v_head_dim),
        )

        return self.o_proj(attn_output)
