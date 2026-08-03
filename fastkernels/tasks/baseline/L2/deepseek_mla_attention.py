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
                 skip_topk: bool = False,
                 topk_indices_buffer: torch.Tensor | None = None,
                 kv_cache_dtype: str | None = None):
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
        # DSA index-topk sharing: a "skip" layer builds NO indexer and reuses the
        # last compute layer's top-k indices from the shared ``topk_indices_buffer``
        # (matches vLLM ``skip_topk``). Compute layers overwrite that buffer.
        self.skip_topk = is_v32 and skip_topk
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = config.index_topk if is_v32 else None

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
            kv_cache_dtype=kv_cache_dtype,
            topk_tokens=self.topk_tokens,
        )
        # Share the kv_b_proj module so ``MLAAttention.forward_impl`` (and
        # the ``fastkernels::unified_mla_attention`` custom op) can project
        # kv_c_normed -> kv without routing a non-tensor arg through the
        # op schema. Bypass ``nn.Module.__setattr__`` to avoid double
        # registration as a submodule.
        object.__setattr__(self.attn, "_kv_b_proj", self.kv_b_proj)

        # DSA Indexer (V3.2 only). Skip layers (DSA index sharing) build no
        # indexer — matching vLLM, whose checkpoint carries indexer weights ONLY
        # for compute layers; building one here would load garbage on skip layers.
        if self.is_v32 and not self.skip_topk:
            _rp = getattr(config, "rope_parameters", None) or {}
            self.indexer = SparseAttnIndexer(
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                n_head=config.index_n_heads,
                head_dim=config.index_head_dim,
                rope_dim=self.qk_rope_head_dim,
                topk_tokens=config.index_topk,
                max_model_len=getattr(config, "max_model_len", 16384),
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
            # Match the main-attention rope: GLM-5.2's "default" rope is plain
            # (bf16 cache, no FlashInfer) whereas DeepSeek-V3.2 YARN keeps
            # FlashInfer + fp32 cache. See ``YarnRotaryEmbedding.is_plain``.
            indexer_is_plain = _rp.get("rope_type") in ('default', 'plain', 'linear')
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
                is_plain=indexer_is_plain,
                # Same as the main rope in ``DeepSeekV3Model``: store the cos/sin
                # cache in the compute dtype ONCE. Without this the cache stays
                # fp32 and ``forward`` re-casts it on every call -- and GLM-5.2's
                # cache is [max_position_embeddings=1048576, 64], i.e. 134 MiB, so
                # each of the ~21 indexer compute layers copied 134 MiB per decode
                # step. That single ``aten::copy_`` was the largest kernel in the
                # whole decode profile (9% of GPU time at bs=1) and has no vLLM
                # counterpart: vLLM's base ``RotaryEmbedding`` also casts at init.
                cache_dtype=getattr(config, "dtype", None),
            )
            # Share the rope_emb module with the indexer so its custom op
            # doesn't need a non-tensor argument (same reasoning as
            # ``self.attn._kv_b_proj`` above).
            object.__setattr__(self.indexer, "_rope_emb", self.indexer_rope_emb)
        else:
            self.indexer = None
            self.indexer_rope_emb = None

    def compute_absorbed_weights(self):
        """Prepare MLA absorbed weights + BF16 indexer wk. Called after weight
        loading but BEFORE FP8 post-processing.

        - BF16 kv_b_proj: builds W_UV [N,L,V] / W_UK_T [N,P,L] now (trivial cast).
        - FP8 kv_b_proj: DEFERS W_UV/W_UK_T to ``finalize_absorbed_weights()``
          (runs AFTER post-processing, via an fp8 GEMM on identity — bit-identical
          to vLLM's ``get_and_maybe_dequant_weights`` use_deep_gemm path).
        - Always: dequantizes the FP8 DSA indexer ``wk`` to BF16 (must happen
          before post-processing, matching vLLM's BF16 indexer wk).
        """
        if hasattr(self.kv_b_proj, 'use_fp8') and self.kv_b_proj.use_fp8:
            # FP8 kv_b_proj: DEFER W_UK/W_UV to finalize_absorbed_weights(),
            # which runs AFTER FP8 post-processing and builds them by running
            # the (DeepGEMM-ready) fp8 kv_b_proj on an identity. That exactly
            # reproduces vLLM's get_and_maybe_dequant_weights ``use_deep_gemm``
            # path (fp8 GEMM on ``torch.eye``), which is BIT-IDENTICAL to vLLM's
            # W_UK_T/W_UV — whereas a direct block-dequant here is ~1-2 bf16 ULP
            # off (it matches ``scaled_dequantize``, but vLLM uses the eye-GEMM
            # on Blackwell). That ULP flows into q_absorbed -> the MLA attention
            # core and flips near-tie tokens. See finalize_absorbed_weights().
            pass
        else:
            # BF16 checkpoint: no postprocessing, dequant is trivial — build now.
            weight = self.kv_b_proj.weight.data.to(torch.bfloat16)
            weight = weight.T  # [L, N*(P+V)]
            L = self.kv_lora_rank
            N = self.num_local_heads
            P = self.qk_nope_head_dim
            V = self.v_head_dim
            weight = weight.view(L, N, P + V)
            W_UK = weight[:, :, :P]  # [L, N, P]
            W_UV = weight[:, :, P:]  # [L, N, V]
            self.attn.W_UV = W_UV.permute(1, 0, 2).contiguous()   # (N, L, V)
            self.attn.W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # (N, P, L)

        # DSA indexer K projection must run in BF16 to match vLLM. vLLM builds
        # the indexer ``wk`` as a BF16 ``MergedColumnParallelLinear`` (fused with
        # ``weights_proj``) and dequantizes any FP8 checkpoint ``wk`` weight to
        # BF16 at load (``_try_load_fp8_indexer_wk`` in deepseek_v2.py). We load
        # ``wk`` as FP8, so dequantize it here — after weight load, before FP8
        # post-processing — so ``k = wk(hidden_states)`` is computed in BF16.
        # Running it in FP8 adds an extra rounding on the indexer K that flows
        # into k_norm -> RoPE -> fp8 K-cache -> MQA logits -> top-k and can flip
        # the selected sparse tokens vs vLLM. Applies to every DSA model
        # (DeepSeek-V3.2 and GLM-5.2); no-op for the BF16 (non-FP8) checkpoints.
        idx = self.indexer
        if idx is not None and getattr(idx.wk, "use_fp8", False):
            wk = idx.wk
            w_bf16 = self._dequant_fp8_block(
                wk.weight.data, wk.weight_scale_inv.data)
            wk.weight = nn.Parameter(w_bf16, requires_grad=False)
            wk.weight.weight_loader = lambda p, w: p.data.copy_(w)
            wk.use_fp8 = False
            # Drop the FP8 apparatus so ``_postprocess_fp8_weights`` (which keys
            # on ``isinstance(m.linear_op, Fp8Linear)``) skips this now-BF16 wk.
            wk.linear_op = None
            if hasattr(wk, "weight_scale_inv"):
                del wk.weight_scale_inv

        # Fuse the (now BF16) ``wk`` and ``weights_proj`` into ONE weight so the
        # indexer forward runs a SINGLE GEMM — exactly vLLM's ``wk_weights_proj``
        # (deepseek_v2.py:636-643, 706-708). Running ``weights_proj`` as a
        # separate GEMM (different N) takes a different cuBLAS accumulation path,
        # giving a ~1-ULP ``weights`` difference that reorders the indexer top-k;
        # the order-sensitive ``flash_mla_sparse_fwd`` then amplifies it at
        # seq>index_topk. Matching vLLM's fused invocation removes that
        # divergence source. Runs for both FP8 (wk just dequantized) and BF16
        # checkpoints; ``wk``/``weights_proj`` are on-device and BF16 here.
        if idx is not None and getattr(idx, "weights_proj", None) is not None:
            idx._wk_wp_fused = torch.cat(
                [idx.wk.weight.data.to(torch.bfloat16),
                 idx.weights_proj.weight.data.to(torch.bfloat16)],
                dim=0,
            ).contiguous()

    def finalize_absorbed_weights(self):
        """Build W_UV / W_UK_T from the POST-PROCESSED fp8 ``kv_b_proj`` via an
        fp8 GEMM on an identity matrix. This reproduces vLLM's
        ``get_and_maybe_dequant_weights`` ``use_deep_gemm`` path
        (``quant_method.apply(layer, eye)``) BIT-FOR-BIT, so the absorbed
        weights — and therefore q_absorbed and the whole MLA attention core —
        are bit-identical to vLLM. Must run AFTER ``_postprocess_fp8_weights``
        (kv_b_proj must be DeepGEMM-ready). No-op for BF16 checkpoints (W_UK_T
        was already built in ``compute_absorbed_weights``)."""
        if self.attn.W_UK_T is not None:
            return
        if not (hasattr(self.kv_b_proj, 'use_fp8') and self.kv_b_proj.use_fp8):
            return
        L = self.kv_lora_rank
        N = self.num_local_heads
        P = self.qk_nope_head_dim
        V = self.v_head_dim
        eye = torch.eye(L, dtype=torch.bfloat16, device=self.kv_b_proj.weight.device)
        with torch.no_grad():
            out = self.kv_b_proj(eye)              # fp8 GEMM(eye) -> [L, N*(P+V)] bf16
            w = out[0] if isinstance(out, tuple) else out
        w = w.view(L, N, P + V)
        W_UK = w[:, :, :P]  # [L, N, P]
        W_UV = w[:, :, P:]  # [L, N, V]
        self.attn.W_UV = W_UV.permute(1, 0, 2).contiguous()   # (N, L, V)
        self.attn.W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # (N, P, L)

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
        if self.is_v32:
            if self.indexer is not None:
                # Compute layer: run the indexer (writes the shared buffer).
                topk_indices = self.indexer(hidden_states, q_c, positions)
            elif self.topk_indices_buffer is not None:
                # Skip layer (DSA index sharing): reuse the last compute layer's
                # top-k indices from the shared buffer — no recompute, matching
                # vLLM's ``skip_topk`` (deepseek_v2.py:1029-1049,1084).
                topk_indices = self.topk_indices_buffer[:N, :self.topk_tokens]

        # MLA attention — kv_b_proj is wired to self.attn._kv_b_proj so
        # the forward takes only tensor args (custom-op safe).
        attn_output = self.attn(
            q, kv_c_normed, k_pe,
            topk_indices=topk_indices,
            output_shape=(N, self.num_local_heads * self.v_head_dim),
        )

        return self.o_proj(attn_output)
