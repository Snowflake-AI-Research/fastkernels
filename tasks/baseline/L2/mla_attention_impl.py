"""MLA attention implementation with BF16 or FP8 paged KV cache.

MLA equivalent of attention_impl.py's Attention class. Handles:

- ``kv_cache_dtype="auto"`` (default, matches stock vLLM): BF16 paged
  KV cache (576 BF16 elems/token = 1152 bytes). Sparse attention calls
  ``flash_mla_sparse_fwd`` directly on the paged cache — no gather or
  upconvert — mirroring ``FlashMLASparseImpl._forward_bf16_kv``.
- ``kv_cache_dtype="fp8_ds_mla"``: FP8 paged KV cache (656 bytes/token).
  Sparse prefill uses a BF16 workspace gathered/upconverted from FP8;
  sparse decode uses ``flash_mla_with_kvcache(..., is_fp8_kvcache=True)``.
- Dense prefill via flash_attn_varlen_func (FA2/FA3, matching vLLM)
- Chunked prefill context: gather from cache, up-project, non-causal attn, merge
- Mixed batch (prefill + decode) with separate FP8/BF16 paths

The default is BF16 so that ``topk_indices``, attention outputs, and
downstream MoE expert assignments are bit-for-bit comparable to vLLM's
stock path. Set ``KB_NANO_KV_CACHE_DTYPE=fp8_ds_mla`` to switch to the
FP8 KV cache (extra memory savings, extra quantization noise).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ....infra.context import get_context
from ..L1.store_kvcache_fp8_mla import (
    StoreKVCacheFP8MLA, GatherKVCacheFP8MLA, GatherAndDequantKVCacheMLA,
)
from ..L1.flash_mla_decode import (
    FlashMLADecode,
    FlashMLADecodeFP8,
    FlashMLAGetMetadata,
    FlashMLAGetMetadataDenseFP8,
)
from ..L1.flash_mla_sparse_prefill import FlashMLASparsePrefill
from ..L1.flash_attn_varlen import FlashAttnVarlen
from ..L1.merge_attn_states import MergeAttnStates
from ..L1.bmm import BatchMatMul
from ..L1.convert_indices import ConvertIndicesToGlobal

_MLA_HEAD_DIM_V = 512
_MLA_WORKSPACE_HEAD_SIZE = 576  # 512 NoPE + 64 RoPE = 576 BF16 dims
MIN_HEADS_FOR_BF16_PREFILL = 32


def _default_kv_cache_dtype() -> str:
    """Resolve the MLA KV cache dtype from ``KB_NANO_KV_CACHE_DTYPE``.

    Defaults to ``"auto"`` (BF16), matching stock vLLM on DeepSeek-V3.2.
    """
    v = os.environ.get("KB_NANO_KV_CACHE_DTYPE", "auto").strip().lower()
    if v in ("", "auto", "bf16", "bfloat16"):
        return "auto"
    if v in ("fp8", "fp8_ds_mla"):
        return "fp8_ds_mla"
    raise ValueError(f"Unsupported KB_NANO_KV_CACHE_DTYPE={v!r}")


def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
    return 64 if num_heads <= 64 else 128


def _compute_prefill_padding() -> int:
    """Mirror vLLM's BF16 sparse prefill padding selection.

    See ``vllm/v1/attention/backends/mla/flashmla_sparse.py:565-568``:
    Hopper (SM90) requires 64-element head padding while Blackwell (SM100)
    requires 128. Older arches default to 64.
    """
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return 64
    return 128 if major == 10 else 64


class MLAAttention(nn.Module):
    """MLA attention with FP8 paged KV cache.

    Unlike standard Attention which has separate k_cache and v_cache,
    MLA uses a single unified cache since kv_c_normed + k_pe are stored together.

    Attributes:
        k_cache, v_cache: both point to the same tensor for engine discovery
        _num_kv_heads: always 1 (MLA = multi-query on the latent)
        _head_dim: kv_lora_rank + qk_rope_head_dim (for cache slot size)
    """

    def __init__(self, num_heads: int, scale: float,
                 qk_nope_head_dim: int, qk_rope_head_dim: int,
                 v_head_dim: int, kv_lora_rank: int,
                 is_sparse: bool = False,
                 kv_cache_dtype: str | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.is_sparse = is_sparse

        if kv_cache_dtype is None:
            kv_cache_dtype = _default_kv_cache_dtype()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla"), (
            f"MLAAttention: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_kv_cache = kv_cache_dtype == "fp8_ds_mla"

        self._num_kv_heads = 1
        # ``_head_dim`` is used by external callers (e.g. the engine) to
        # size the cache. For BF16 it's the packed
        # ``kv_lora_rank + qk_rope_head_dim`` (576). For FP8 the cache is
        # uint8 with 656 bytes/token, which we continue to advertise here.
        self._head_dim = (
            kv_lora_rank + qk_rope_head_dim if not self.use_fp8_kv_cache else 656
        )

        self.k_cache = self.v_cache = torch.tensor([])

        self.fp8_decode_padded_heads = _compute_fp8_decode_padded_heads(num_heads)
        # BF16 sparse prefill kernel head-pad: 64 on Hopper, 128 on Blackwell
        # (matches vLLM's ``FlashMLASparseImpl.prefill_padding``).
        self.prefill_padding = _compute_prefill_padding()

        # W_UV: absorbed V projection from kv_b_proj, computed after weight loading.
        # Shape: [num_heads, kv_lora_rank, v_head_dim]
        self.W_UV: torch.Tensor | None = None
        # W_UK_T: absorbed K projection transposed, for decode query absorption.
        # Shape: [num_heads, qk_nope_head_dim, kv_lora_rank]
        self.W_UK_T: torch.Tensor | None = None

        self.store_kvcache = StoreKVCacheFP8MLA(kv_cache_dtype=kv_cache_dtype)
        self.gather_kvcache = GatherKVCacheFP8MLA()
        self.gather_dequant_kvcache = GatherAndDequantKVCacheMLA()
        self.decode_op = FlashMLADecode()
        # Dense FP8 decode entry-point (matches vLLM's
        # ``flash_mla_with_kvcache_fp8`` path used in
        # ``vllm/v1/attention/backends/mla/flashmla.py``). Falls back to the
        # generic ``FlashMLADecode`` with ``is_fp8_kvcache=True`` when the
        # specialized kernel is not available (older vLLM builds).
        self.decode_op_fp8 = FlashMLADecodeFP8()
        self.sparse_prefill_op = FlashMLASparsePrefill()
        self.get_metadata = FlashMLAGetMetadata()
        self.get_metadata_dense_fp8 = FlashMLAGetMetadataDenseFP8()
        self.varlen_attn = FlashAttnVarlen()
        self.merge_states = MergeAttnStates()
        self.bmm = BatchMatMul()
        self.convert_indices = ConvertIndicesToGlobal()

        # Per-layer dequant scales for FP8 dense MLA decode. vLLM populates
        # these via ``maybe_calc_kv_scales`` (currently 1.0 unless calibrated).
        # We mirror the ``layer._q_scale`` / ``layer._k_scale`` buffers from
        # ``vllm/model_executor/layers/attention/attention.py:95-100``.
        self.register_buffer(
            "_q_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

        # Custom-op dispatch scaffolding (matches Attention L2 module):
        # ``_use_custom_op`` is flipped to True by ``enable_custom_ops`` once
        # the model is wrapped with ``torch.compile``. ``_layer_name`` is
        # populated by ``auto_register_no_compile_layers``.
        self._use_custom_op = False
        self._layer_name = ""
        # Reference to the enclosing ``kv_b_proj`` module, set by the parent
        # ``DeepSeekMLAAttention``. Stored via ``object.__setattr__`` at the
        # parent site to avoid double-registration as an ``nn.Module``
        # submodule (which would shadow parent weights). ``None`` until
        # wired up.
        self._kv_b_proj: nn.Module | None = None

    def forward(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor, kv_b_proj: nn.Module | None = None,
                topk_indices: torch.Tensor | None = None,
                output_shape: tuple | None = None) -> torch.Tensor:
        # Keep the historical positional ``kv_b_proj`` argument for direct
        # (eager / unit-test) callers but prefer the stored reference so the
        # torch.compile custom-op path only has tensor-typed arguments.
        if kv_b_proj is not None and self._kv_b_proj is None:
            object.__setattr__(self, "_kv_b_proj", kv_b_proj)

        # ``output_shape`` is intentionally ignored on the dispatch path: the
        # output is always reshaped to ``(N, num_heads * v_head_dim)`` where
        # ``N = q.shape[0]``. Computing it from ``q`` keeps the batch dim
        # symbolic under torch.compile (passing a precomputed ``int[]`` here
        # would force Dynamo to specialize ``q.shape[0]`` to a constant).
        if self._use_custom_op:
            return torch.ops.kb_nano.unified_mla_attention(
                q, kv_c_normed, k_pe, topk_indices, self._layer_name,
            )
        return self.forward_impl(q, kv_c_normed, k_pe, topk_indices)

    def forward_impl(self, q: torch.Tensor, kv_c_normed: torch.Tensor,
                     k_pe: torch.Tensor,
                     topk_indices: torch.Tensor | None = None) -> torch.Tensor:
        ctx = get_context()
        N = q.shape[0]

        kv_cache = self.k_cache
        kv_b_proj = self._kv_b_proj
        assert kv_b_proj is not None, "MLAAttention._kv_b_proj is not wired"

        if kv_cache.numel() and ctx.slot_mapping is not None:
            self.store_kvcache(kv_c_normed, k_pe, kv_cache, ctx.slot_mapping)

        if self.is_sparse and topk_indices is not None and kv_cache.ndim >= 2:
            o = self._forward_sparse(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices)
        elif ctx.is_mixed:
            o = self._forward_mixed(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)
        else:
            o = self._forward_pure(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)

        return o.view(N, self.num_heads * self.v_head_dim)

    def _forward_pure(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        if ctx.is_prefill:
            return self._forward_mha(q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx)
        return self._forward_dense_decode(q, kv_cache, ctx)

    def _run_prefill_new_tokens(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                                max_seqlen_q, max_seqlen_k,
                                return_softmax_lse=False):
        """Run causal attention on new prefill tokens via the L1
        FlashAttnVarlen op."""
        attn_out = self.varlen_attn(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )
        if isinstance(attn_out, tuple):
            return attn_out[0], attn_out[1]
        if return_softmax_lse:
            return attn_out, None
        return attn_out

    def _run_prefill_context_chunk(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k):
        """Run non-causal attention on a context chunk via the L1
        FlashAttnVarlen op (always returns LSE for merging)."""
        attn_out = self.varlen_attn(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )
        if isinstance(attn_out, tuple):
            return attn_out[0], attn_out[1]
        return attn_out, None

    def _concat_k_nope_k_pe(self, k_nope, k_pe):
        """Concatenate k_nope and expanded k_pe along the head_dim."""
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype, device=k_nope.device,
        )
        k[..., :k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1]:] = k_pe
        return k

    def _compute_prefill_context(self, q, kv_cache, kv_b_proj, ctx):
        """Gather cached context, up-project, run non-causal attn, merge chunks.

        Matches vllm's MLACommonImpl._compute_prefill_context:
        for each context chunk, gather from FP8 cache into BF16 workspace,
        split into kv_c_normed and k_pe, project kv_c_normed through kv_b_proj
        to get k_nope and v, run non-causal attention, merge with
        merge_attn_states.
        """
        chunked_ctx = ctx.chunked_context
        assert chunked_ctx is not None

        output = None
        output_lse = None
        iters = len(chunked_ctx.seq_tot)
        workspace = chunked_ctx.workspace

        if ctx.is_mixed:
            query_start_loc = ctx.prefill_cu_seqlens_q
            max_query_len = ctx.prefill_max_seqlen_q
        else:
            query_start_loc = ctx.cu_seqlens_q
            max_query_len = ctx.max_seqlen_q

        for i in range(iters):
            toks = chunked_ctx.seq_tot[i]

            block_table = (
                ctx.prefill_block_tables if ctx.is_mixed else ctx.block_tables
            )

            if GatherAndDequantKVCacheMLA.available:
                self.gather_dequant_kvcache(
                    kv_cache, workspace, block_table,
                    chunked_ctx.cu_seq_lens[i],
                    chunked_ctx.token_to_seq[i],
                    chunked_ctx.chunk_total_token[i],
                    chunked_ctx.starts[i],
                )
            else:
                self.gather_kvcache(
                    kv_cache, block_table,
                    chunked_ctx.cu_seq_lens[i],
                    chunked_ctx.starts[i],
                    chunked_ctx.cu_seq_lens[i].shape[0] - 1,
                    workspace,
                )

            kv_c_normed = workspace[:toks, :self.kv_lora_rank]
            k_pe = workspace[:toks, self.kv_lora_rank:].unsqueeze(1)

            kv_nope = kv_b_proj(kv_c_normed)
            kv_nope = kv_nope.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk(
                q=q, k=k, v=v,
                cu_seqlens_q=query_start_loc,
                cu_seqlens_k=chunked_ctx.cu_seq_lens[i],
                max_seqlen_q=max_query_len,
                max_seqlen_k=chunked_ctx.max_seq_lens[i],
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                self.merge_states(
                    output=output_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                    output_lse=output_lse_tmp,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_mha(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        """Dense prefill with chunked context support (matches vllm forward_mha)."""
        N = q.shape[0]
        has_context = ctx.chunked_context is not None

        kv = kv_b_proj(kv_c_normed)
        kv = kv.view(N, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        if ctx.is_mixed:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
            max_sq = ctx.prefill_max_seqlen_q
            max_sk = ctx.prefill_max_seqlen_k
        else:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k
            max_sq = ctx.max_seqlen_q
            max_sk = ctx.max_seqlen_k

        output_prefill = self._run_prefill_new_tokens(
            q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_q,
            max_seqlen_q=max_sq, max_seqlen_k=max_sq,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output_prefill
            context_output, context_lse = self._compute_prefill_context(
                q, kv_cache, kv_b_proj, ctx)

            output = torch.empty(N, self.num_heads, self.v_head_dim,
                                 dtype=q.dtype, device=q.device)
            self.merge_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output[..., :self.v_head_dim],
                suffix_lse=suffix_lse,
            )
            return output.reshape(N, self.num_heads * self.v_head_dim)
        else:
            o = output_prefill
            if isinstance(o, tuple):
                o = o[0]
            return o.reshape(N, self.num_heads * self.v_head_dim)

    def _v_up_proj(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Project FlashMLA output from kv_lora_rank to v_head_dim per head.

        Matches vllm's MLAAttention._v_up_proj: (B, N, L) -> (N, B, L) x
        (N, L, V) -> (N, B, V) -> (B, N*V).
        """
        if self.W_UV is None:
            return attn_out[..., :self.v_head_dim]
        N = attn_out.shape[0]
        o = attn_out.view(N, self.num_heads, self.kv_lora_rank)
        o = o.transpose(0, 1)  # (N, B, L)
        out = self.bmm(o, self.W_UV)  # (N, B, V)
        return out.transpose(0, 1).reshape(N, self.num_heads * self.v_head_dim)

    def _forward_dense_decode(self, q, kv_cache, ctx):
        cache_seqlens = ctx.context_lens
        block_table = ctx.block_tables
        if not self.use_fp8_kv_cache:
            q = self._absorb_q_to_latent(q)
            q = q.unsqueeze(1)
            tile_sched_meta, _ = self.get_metadata(
                cache_seqlens, self.num_heads, num_heads_k=1,
            )
            o, _ = self.decode_op(
                q,
                kv_cache.unsqueeze(-2),
                block_table,
                cache_seqlens,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                causal=True,
            )
            o = o.reshape(-1, o.shape[-2], o.shape[-1])
            return self._v_up_proj(o)

        # Mirrors vLLM's dense FP8 MLA decode path
        # (vllm/v1/attention/backends/mla/flashmla.py:289-302):
        # use the specialized ``flash_mla_with_kvcache_fp8`` kernel with
        # per-layer ``descale_q`` / ``descale_k`` and ``causal=True``.
        if self.decode_op_fp8.available:
            tile_sched_meta, num_splits = self.get_metadata_dense_fp8(
                cache_seqlens, self.num_heads, num_heads_k=1,
            )
            o, _ = self.decode_op_fp8(
                q.unsqueeze(1), kv_cache.view(torch.uint8).unsqueeze(-2),
                block_table, cache_seqlens,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                num_splits=num_splits,
                softmax_scale=self.scale,
                causal=True,
                descale_q=self._q_scale.reshape(1),
                descale_k=self._k_scale.reshape(1),
            )
            o = o.reshape(-1, o.shape[-2], o.shape[-1])
            return self._v_up_proj(o)

        # Fallback: generic kernel with is_fp8_kvcache=True (older vLLM).
        tile_sched_meta, _ = self.get_metadata(
            cache_seqlens, self.num_heads, num_heads_k=1,
            is_fp8_kvcache=True)
        o, _ = self.decode_op(
            q.unsqueeze(1), kv_cache.view(torch.uint8).unsqueeze(-2),
            block_table, cache_seqlens,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.scale,
            causal=True,
            is_fp8_kvcache=True,
        )
        o = o.reshape(-1, o.shape[-2], o.shape[-1])
        return self._v_up_proj(o)

    def _forward_sparse(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices):
        """Sparse attention dispatcher.

        Mirrors ``vllm/v1/attention/backends/mla/flashmla_sparse.py``'s
        ``FlashMLASparseImpl.forward_mqa`` selection:

        * ``kv_cache_dtype="auto"`` (BF16): **single kernel path** for
          both prefill and decode — ``flash_mla_sparse_fwd`` reads the
          BF16 paged cache directly. Matches vLLM's ``_forward_bf16_kv``.
        * ``kv_cache_dtype="fp8_ds_mla"``:

          * mixed-batch FP8 path (``num_heads < MIN_HEADS_FOR_BF16_PREFILL``):
            one ``flash_mla_with_kvcache`` call for all tokens.
          * separate prefill / decode FP8 path (large head count): BF16
            workspace prefill + FP8 decode kernel.
          * pure FP8 decode path.
        """
        N = q.shape[0]

        if not self.use_fp8_kv_cache:
            return self._forward_sparse_bf16(q, kv_cache, ctx, topk_indices)

        use_mixed_batch = self.num_heads < MIN_HEADS_FOR_BF16_PREFILL

        if ctx.is_prefill:
            num_pf, num_dc = N, 0
            is_mixed_or_prefill = True
        elif ctx.is_mixed:
            num_pf = ctx.num_prefill_tokens
            num_dc = ctx.num_decode_tokens
            is_mixed_or_prefill = True
        else:
            return self._forward_sparse_decode(q, kv_cache, ctx, topk_indices)

        if is_mixed_or_prefill and use_mixed_batch:
            return self._forward_sparse_mixed_batch(
                q, kv_cache, ctx, topk_indices,
                num_prefill_tokens=num_pf,
                num_decode_tokens=num_dc,
            )
        return self._forward_sparse_separate(
            q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx, topk_indices,
            num_prefill_tokens=num_pf, num_decode_tokens=num_dc)

    def _forward_sparse_bf16(self, q, kv_cache, ctx, topk_indices):
        """BF16 KV cache sparse path, identical to vLLM's ``_forward_bf16_kv``.

        All tokens (prefill, decode, mixed) go through a single
        ``flash_mla_sparse_fwd`` call over the BF16 paged cache:

        * convert per-request ``topk_indices`` into global slot indices
          (``convert_indices`` with the batch's unified block table);
        * absorb ``q`` through ``W_UK_T`` and concat with ``q_pe`` to get
          a 576-D head ("MQA 576/512 approach");
        * pad the head count to ``self.prefill_padding`` (64 on Hopper /
          128 on Blackwell) as required by the BF16 sparse kernel;
        * call ``flash_mla_sparse_fwd(q, kv_cache.view(-1, 1, 576),
          topk_indices.view(N, 1, topk), sm_scale)``;
        * slice output heads back to ``num_heads`` and up-project via
          ``W_UV``.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])
        num_heads = self.num_heads
        pad_h = self.prefill_padding

        # Build the unified block_table (decode rows first, prefill after)
        # and per-token req_ids so ``convert_indices`` can translate
        # per-request topk indices to global slot indices of ``kv_cache``.
        if ctx.is_mixed:
            num_dc = ctx.num_decode_tokens
            num_pf = ctx.num_prefill_tokens
            num_decode_seqs = (
                ctx.decode_block_tables.shape[0]
                if ctx.decode_block_tables is not None and num_dc > 0 else 0
            )
            num_prefill_seqs = (
                ctx.prefill_block_tables.shape[0]
                if ctx.prefill_block_tables is not None and num_pf > 0 else 0
            )
            if num_decode_seqs > 0 and num_prefill_seqs > 0:
                d_bt = ctx.decode_block_tables
                p_bt = ctx.prefill_block_tables
                max_b = max(d_bt.shape[1], p_bt.shape[1])
                if d_bt.shape[1] < max_b:
                    pad = torch.full((d_bt.shape[0], max_b - d_bt.shape[1]),
                                     -1, dtype=d_bt.dtype, device=d_bt.device)
                    d_bt = torch.cat([d_bt, pad], dim=1)
                if p_bt.shape[1] < max_b:
                    pad = torch.full((p_bt.shape[0], max_b - p_bt.shape[1]),
                                     -1, dtype=p_bt.dtype, device=p_bt.device)
                    p_bt = torch.cat([p_bt, pad], dim=1)
                unified_block_table = torch.cat([d_bt, p_bt], dim=0)
            elif num_prefill_seqs > 0:
                unified_block_table = ctx.prefill_block_tables
            elif num_decode_seqs > 0:
                unified_block_table = ctx.decode_block_tables
            else:
                return torch.zeros(
                    N, num_heads * self.v_head_dim,
                    dtype=q.dtype, device=q.device,
                )
        else:
            if ctx.block_tables is None:
                return torch.zeros(
                    N, num_heads * self.v_head_dim,
                    dtype=q.dtype, device=q.device,
                )
            unified_block_table = ctx.block_tables
            num_dc = 0 if ctx.is_prefill else N
            num_pf = N if ctx.is_prefill else 0
            num_decode_seqs = (
                0 if ctx.is_prefill else unified_block_table.shape[0]
            )
            num_prefill_seqs = (
                unified_block_table.shape[0] if ctx.is_prefill else 0
            )

        req_ids = ctx.req_id_per_token
        if req_ids is None:
            req_ids = torch.zeros(N, dtype=torch.int32, device=q.device)
            if ctx.is_mixed:
                for i in range(num_decode_seqs):
                    req_ids[i] = i
                pf_cu_q = ctx.prefill_cu_seqlens_q
                if pf_cu_q is not None:
                    for r in range(num_prefill_seqs):
                        qs = int(pf_cu_q[r].item()) + num_dc
                        qe = int(pf_cu_q[r + 1].item()) + num_dc
                        req_ids[qs:qe] = num_decode_seqs + r
            else:
                cu_q = ctx.cu_seqlens_q
                if cu_q is not None:
                    nseqs = (
                        num_prefill_seqs if num_prefill_seqs > 0
                        else num_decode_seqs
                    )
                    for r in range(nseqs):
                        qs = int(cu_q[r].item())
                        qe = int(cu_q[r + 1].item())
                        req_ids[qs:qe] = r

        topk_global = self.convert_indices(
            topk_indices, unified_block_table, block_size, req_ids=req_ids,
        )

        # Absorb q into the 576-D latent space.
        q_latent = self._absorb_q_to_latent(q)  # [N, H, 576]

        # Pad heads to multiple of ``prefill_padding`` (BF16 sparse kernel
        # requirement, see ``vllm/v1/attention/backends/mla/flashmla_sparse
        # .py:_bf16_flash_mla_kernel``).
        actual_heads = q_latent.shape[1]
        if actual_heads % pad_h != 0:
            assert pad_h % actual_heads == 0
            q_pad = q_latent.new_empty(N, pad_h, q_latent.shape[2])
            q_pad[:, :actual_heads, :] = q_latent
            q_latent = q_pad

        # View the BF16 paged cache as (num_blocks * block_size, 1, 576)
        # so ``flash_mla_sparse_fwd`` can gather by global slot index.
        kv_flat = kv_cache.view(-1, 1, kv_cache.shape[-1])
        topk_3d = topk_global.view(N, 1, -1)

        out = self.sparse_prefill_op(q_latent, kv_flat, topk_3d, self.scale)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # Trim padded heads back to num_heads.
        out = out[:, :num_heads, :]
        return self._v_up_proj(out)

    def _absorb_q_to_latent(self, q: torch.Tensor) -> torch.Tensor:
        """Absorb q_nope through W_UK_T into the latent space and concat q_pe.

        Output shape: ``[..., H, kv_lora_rank + qk_rope_head_dim]`` (576 for
        DeepSeek-V3.2). Matches vLLM's MLA decode/sparse query absorption.
        """
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim:]
        # (H, N, P) @ (H, P, L) -> (H, N, L) -> (N, H, L)
        q_absorbed = self.bmm(
            q_nope.transpose(0, 1), self.W_UK_T,
        ).transpose(0, 1)
        return torch.cat([q_absorbed, q_pe], dim=-1)

    def _forward_sparse_mixed_batch(self, q, kv_cache, ctx, topk_indices,
                                    num_prefill_tokens, num_decode_tokens):
        """Mixed-batch FP8 sparse path (vLLM's ``_forward_fp8_kv_mixed_batch``).

        All tokens are treated as one logical batch of length ``T = N``,
        ``B = 1``, ``H = padded_heads``. This avoids the BF16 prefill kernel's
        head padding overhead and exactly matches what vLLM uses when
        ``num_heads < MIN_HEADS_FOR_BF16_PREFILL`` (e.g. TP=8, 16 heads).

        Mirrors ``vllm/v1/attention/backends/mla/flashmla_sparse.py:
        _forward_fp8_kv_mixed_batch``.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])

        # Mixed-batch always sources K from the paged FP8 cache (prefill tokens
        # have already been written via ``store_kvcache`` before attention).
        # The triton kernel only needs the per-token req_id + the full
        # block_table — workspace_starts are unused (HAS_PREFILL=False branch).
        req_ids = ctx.req_id_per_token
        if req_ids is None:
            req_ids = torch.arange(N, dtype=torch.int32, device=q.device)
        block_table = ctx.block_tables

        topk_global = self.convert_indices(
            topk_indices, block_table, block_size, req_ids=req_ids,
        )

        # Absorb q into the 576-D latent space.
        q_latent = self._absorb_q_to_latent(q)  # [N, H, 576]

        # Pad heads to 64 or 128 (FP8 sparse decode kernel requirement).
        # Reshape to (B=1, T=N, H, D) and pad along the head dim.
        q_4d = q_latent.unsqueeze(0)  # (1, N, H, 576)
        q_4d, actual_heads = self._pad_q_for_fp8(q_4d)
        padded_heads = q_4d.shape[-2]
        topk_3d = topk_global.unsqueeze(0)  # (1, N, topk)

        # Single-batch metadata (matches vLLM's ``_build_fp8_mixed_decode_prefill``).
        topk = topk_indices.shape[-1]
        topk_tensor = torch.tensor(
            [topk], dtype=torch.int32, device=q.device,
        )
        dummy_bt = torch.empty(
            (1, 1), dtype=torch.int32, device=q.device,
        )
        # Single "sequence" containing all N tokens with padded_heads queries each.
        tile_sched_meta, _ = self.get_metadata(
            topk_tensor, N * padded_heads,
            topk=topk, num_heads_q=padded_heads,
            num_heads_k=1, is_fp8_kvcache=True,
        )

        o, _ = self.decode_op(
            q_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
            dummy_bt, topk_tensor,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_3d,
        )

        # (1, N, padded_heads, 512) -> (N, num_heads, 512)
        o = o.view(N, padded_heads, o.shape[-1])
        if actual_heads < padded_heads:
            o = o[:, :actual_heads, :]
        return self._v_up_proj(o)

    def _pad_q_for_fp8(self, q: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad num_heads to 64 or 128 as required by the FP8 sparse decode kernel."""
        actual_heads = q.shape[-2]
        padded_heads = self.fp8_decode_padded_heads
        if actual_heads >= padded_heads:
            return q, actual_heads
        pad_shape = list(q.shape)
        pad_shape[-2] = padded_heads
        q_padded = q.new_zeros(pad_shape)
        q_padded[..., :actual_heads, :] = q
        return q_padded, actual_heads

    def _forward_sparse_decode(self, q, kv_cache, ctx, topk_indices):
        """Sparse FP8 decode: absorb q into latent space, then FlashMLA sparse.

        The sparse decode kernel requires head_size_k == 576 (kv_lora_rank +
        qk_rope_head_dim). We absorb q_nope via W_UK_T: (N,H,P)@(H,P,L) →
        (N,H,L), then concatenate with q_pe to get (N,H,576).
        Matches vllm's MLACommonImpl decode query absorption.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])
        num_decodes = ctx.block_tables.shape[0]

        req_ids = ctx.req_id_per_token
        if req_ids is None:
            req_ids = torch.arange(N, dtype=torch.int32, device=q.device)

        topk_indices = self.convert_indices(
            topk_indices, ctx.block_tables, block_size, req_ids=req_ids)

        # Absorb q_nope into latent space via W_UK_T
        q_nope = q[..., :self.qk_nope_head_dim]   # [N, H, P]
        q_pe = q[..., self.qk_nope_head_dim:]      # [N, H, rope]

        # (H, N, P) @ (H, P, L) -> (H, N, L) -> (N, H, L)
        q_absorbed = self.bmm(
            q_nope.transpose(0, 1), self.W_UK_T,
        ).transpose(0, 1)

        # Concat absorbed nope + rope -> [N, H, L+rope=576]
        q_latent = torch.cat([q_absorbed, q_pe], dim=-1)

        decode_query_len = N // num_decodes if num_decodes > 0 else N
        q_4d = q_latent.view(num_decodes, decode_query_len, self.num_heads, q_latent.shape[-1])
        topk_4d = topk_indices.view(num_decodes, decode_query_len, -1)

        q_4d, actual_heads = self._pad_q_for_fp8(q_4d)
        padded_heads = q_4d.shape[-2]

        topk = topk_indices.shape[-1]
        topk_tensor = torch.full(
            (num_decodes,), topk, dtype=torch.int32, device=q.device)
        dummy_bt = torch.empty(
            (num_decodes, 1), dtype=torch.int32, device=q.device)

        tile_sched_meta, _ = self.get_metadata(
            topk_tensor, decode_query_len * padded_heads,
            topk=topk, num_heads_q=padded_heads,
            num_heads_k=1, is_fp8_kvcache=True)

        o, _ = self.decode_op(
            q_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
            dummy_bt, topk_tensor,
            head_dim_v=_MLA_HEAD_DIM_V,
            tile_scheduler_metadata=tile_sched_meta,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_4d,
        )

        o = o.view(-1, padded_heads, o.shape[-1])
        if actual_heads < padded_heads:
            o = o[:, :actual_heads, :]
        return self._v_up_proj(o)

    def _forward_sparse_separate(self, q, kv_c_normed, k_pe, kv_b_proj,
                                 kv_cache, ctx, topk_indices,
                                 num_prefill_tokens, num_decode_tokens):
        """Separate prefill (BF16 workspace) and decode (FP8 kernel).

        Mirrors vLLM's ``_forward_fp8_kv_separate_prefill_decode``: ALL
        tokens flow through sparse attention (workspace-gather for prefill,
        direct FP8 paged decode for decode tokens). We must NOT fall back
        to the dense ``_forward_mha`` path here — that bypasses the DSA
        top-k indices entirely and produces dense full attention output.
        """
        N = q.shape[0]
        block_size = int(kv_cache.shape[1])

        # ``ctx.block_tables`` is only populated by the pure-prefill /
        # pure-decode set_context paths.  ``prepare_mixed_batch`` populates
        # ``prefill_block_tables`` and ``decode_block_tables`` separately
        # (vLLM keeps a single unified block_table covering all sequences).
        # Pick the right table(s) so we have something to feed to
        # ``convert_indices`` and to derive ``num_seqs_total``.
        if ctx.is_mixed:
            num_decode_seqs = (
                ctx.decode_block_tables.shape[0]
                if ctx.decode_block_tables is not None and num_decode_tokens > 0
                else 0
            )
            num_prefill_seqs = (
                ctx.prefill_block_tables.shape[0]
                if ctx.prefill_block_tables is not None
                and num_prefill_tokens > 0
                else 0
            )
            num_seqs_total = num_decode_seqs + num_prefill_seqs

            # Build a unified block_table for ``convert_indices``: rows
            # ``[0:num_decode_seqs]`` are decode requests, rows
            # ``[num_decode_seqs:]`` are prefill requests. For pure
            # prefill (no decode) this is just ``prefill_block_tables``.
            if num_decode_seqs > 0 and num_prefill_seqs > 0:
                # Pad to common max_blocks before concatenating.
                d_bt = ctx.decode_block_tables
                p_bt = ctx.prefill_block_tables
                max_b = max(d_bt.shape[1], p_bt.shape[1])
                if d_bt.shape[1] < max_b:
                    pad = torch.full(
                        (d_bt.shape[0], max_b - d_bt.shape[1]),
                        -1, dtype=d_bt.dtype, device=d_bt.device,
                    )
                    d_bt = torch.cat([d_bt, pad], dim=1)
                if p_bt.shape[1] < max_b:
                    pad = torch.full(
                        (p_bt.shape[0], max_b - p_bt.shape[1]),
                        -1, dtype=p_bt.dtype, device=p_bt.device,
                    )
                    p_bt = torch.cat([p_bt, pad], dim=1)
                unified_block_table = torch.cat([d_bt, p_bt], dim=0)
            elif num_prefill_seqs > 0:
                unified_block_table = ctx.prefill_block_tables
            elif num_decode_seqs > 0:
                unified_block_table = ctx.decode_block_tables
            else:
                # Nothing to do.
                return torch.zeros(
                    N, self.num_heads * self.v_head_dim,
                    dtype=q.dtype, device=q.device,
                )
        else:
            if ctx.block_tables is None:
                # Truly nothing to attend to — return zeros (matches an
                # empty MLA call).
                return torch.zeros(
                    N, self.num_heads * self.v_head_dim,
                    dtype=q.dtype, device=q.device,
                )
            unified_block_table = ctx.block_tables
            num_seqs_total = unified_block_table.shape[0]
            num_decode_seqs = getattr(
                ctx, 'num_decode_seqs',
                num_seqs_total if num_decode_tokens > 0 else 0,
            )
            num_prefill_seqs = num_seqs_total - num_decode_seqs

        req_ids = ctx.req_id_per_token
        if req_ids is None:
            # Per-token request id. For mixed batch:
            #   decode tokens ``[0:num_decode_tokens]`` -> request 0..num_dc_seqs-1
            #   prefill tokens ``[num_decode_tokens:]`` derived from prefill_cu_q
            # For pure prefill (single sequence, our diagnostic case) all
            # tokens belong to request 0; the previous implementation used
            # ``arange`` which over-indexed the block_table for any
            # single-sequence prefill > 1 token. Build req_ids correctly
            # from the cumulative seqlens metadata.
            req_ids = torch.zeros(N, dtype=torch.int32, device=q.device)
            if ctx.is_mixed:
                # Decode rows come first.
                for i in range(num_decode_seqs):
                    req_ids[i] = i
                pf_cu_q = ctx.prefill_cu_seqlens_q
                if pf_cu_q is not None:
                    for r in range(num_prefill_seqs):
                        qs = int(pf_cu_q[r].item()) + num_decode_tokens
                        qe = int(pf_cu_q[r + 1].item()) + num_decode_tokens
                        # Decode block_table sits at rows [0:num_decode_seqs];
                        # prefill rows are appended after, hence + offset.
                        req_ids[qs:qe] = num_decode_seqs + r
            else:
                cu_q = ctx.cu_seqlens_q
                if cu_q is not None:
                    for r in range(num_prefill_seqs if num_prefill_seqs > 0 else num_decode_seqs):
                        qs = int(cu_q[r].item())
                        qe = int(cu_q[r + 1].item())
                        req_ids[qs:qe] = r

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill = num_prefill_tokens > 0

        if has_prefill:
            if ctx.is_mixed:
                pf_bt = (
                    ctx.prefill_block_tables
                    if ctx.prefill_block_tables is not None
                    else unified_block_table[num_decode_seqs:]
                )
                pf_cu = ctx.prefill_cu_seqlens_k
                pf_seq_lens = pf_cu[1:] - pf_cu[:-1]
            else:
                pf_bt = unified_block_table
                pf_cu = ctx.cu_seqlens_k
                pf_seq_lens = pf_cu[1:] - pf_cu[:-1]

            prefill_request_ids = torch.full((N,), -1, dtype=torch.int32, device=q.device)
            prefill_workspace_starts = torch.zeros(num_prefill_seqs, dtype=torch.int32, device=q.device)

            if num_prefill_seqs > 1:
                prefill_workspace_starts[1:] = torch.cumsum(pf_seq_lens[:-1], dim=0).int()

            if ctx.is_mixed:
                pf_cu_q = ctx.prefill_cu_seqlens_q
                for req_idx in range(num_prefill_seqs):
                    qs = int(pf_cu_q[req_idx].item()) + num_decode_tokens
                    qe = int(pf_cu_q[req_idx + 1].item()) + num_decode_tokens
                    prefill_request_ids[qs:qe] = req_idx
            else:
                cu_q = ctx.cu_seqlens_q
                for req_idx in range(num_prefill_seqs):
                    qs = int(cu_q[req_idx].item())
                    qe = int(cu_q[req_idx + 1].item())
                    prefill_request_ids[qs:qe] = req_idx

        topk_global = self.convert_indices(
            topk_indices, unified_block_table, block_size,
            req_ids=req_ids,
            prefill_request_ids=prefill_request_ids,
            prefill_workspace_starts=prefill_workspace_starts,
        )

        # Absorb q into the 576-D latent space (mirrors what vLLM's
        # MLAAttention.forward_impl does *before* calling
        # ``forward_mqa``).  Both the BF16 sparse-prefill kernel and the
        # FP8 sparse-decode kernel expect ``q`` with head-dim
        # ``kv_lora_rank + qk_rope_head_dim`` (576 for V3.2), not the
        # un-absorbed 192-D layout produced by ``q_b_proj``.
        q = self._absorb_q_to_latent(q)  # [N, H, kv_lora_rank+rope]

        out = torch.empty(N, self.num_heads, self.kv_lora_rank,
                          dtype=q.dtype, device=q.device)

        if num_decode_tokens > 0:
            nd = num_decode_tokens
            q_dc = q[:nd]
            topk_dc = topk_global[:nd]
            num_decodes = num_decode_seqs

            q_dc_4d = q_dc.view(num_decodes, -1, self.num_heads, q.shape[-1])
            topk_dc_4d = topk_dc.view(num_decodes, -1, topk_dc.shape[-1])
            q_dc_4d, actual_heads = self._pad_q_for_fp8(q_dc_4d)
            padded_heads = q_dc_4d.shape[-2]
            decode_query_len = q_dc_4d.shape[1]

            topk = topk_dc.shape[-1]
            topk_tensor = torch.full(
                (num_decodes,), topk, dtype=torch.int32, device=q.device)
            dummy_bt = torch.empty(
                (num_decodes, 1), dtype=torch.int32, device=q.device)

            tile_sched_meta, _ = self.get_metadata(
                topk_tensor, decode_query_len * padded_heads,
                topk=topk, num_heads_q=padded_heads,
                num_heads_k=1, is_fp8_kvcache=True)

            o_dc, _ = self.decode_op(
                q_dc_4d, kv_cache.view(torch.uint8).unsqueeze(-2),
                dummy_bt, topk_tensor,
                head_dim_v=_MLA_HEAD_DIM_V,
                tile_scheduler_metadata=tile_sched_meta,
                softmax_scale=self.scale,
                is_fp8_kvcache=True,
                indices=topk_dc_4d,
            )
            o_dc = o_dc.view(-1, padded_heads, o_dc.shape[-1])
            if actual_heads < padded_heads:
                o_dc = o_dc[:, :actual_heads, :]
            out[:nd] = o_dc

        if num_prefill_tokens > 0:
            np_ = num_prefill_tokens
            q_pf = q[num_decode_tokens:] if ctx.is_mixed else q
            topk_pf = topk_global[num_decode_tokens:] if ctx.is_mixed else topk_global

            total_seq_len = int(pf_seq_lens.sum().item())
            workspace = torch.empty(total_seq_len, _MLA_WORKSPACE_HEAD_SIZE,
                                    dtype=torch.bfloat16, device=q.device)
            self.gather_kvcache(
                kv_cache, pf_bt, pf_seq_lens,
                prefill_workspace_starts, num_prefill_seqs, workspace,
            )

            workspace_kv = workspace.view(-1, 1, _MLA_WORKSPACE_HEAD_SIZE)

            prefill_padding = self.prefill_padding
            actual_h = q_pf.shape[1]
            q_pf_3d = q_pf
            if actual_h % prefill_padding != 0:
                pad_h = prefill_padding
                q_padded = q_pf_3d.new_empty(q_pf_3d.shape[0], pad_h, q_pf_3d.shape[2])
                q_padded[:, :actual_h, :] = q_pf_3d
                q_pf_3d = q_padded

            topk_pf_3d = topk_pf.view(np_, 1, -1)
            pf_out = self.sparse_prefill_op(
                q_pf_3d, workspace_kv, topk_pf_3d, self.scale)

            if isinstance(pf_out, (tuple, list)):
                pf_out = pf_out[0]
            pf_out = pf_out[:, :actual_h, :]

            if ctx.is_mixed:
                out[num_decode_tokens:] = pf_out
            else:
                out[:] = pf_out

        return self._v_up_proj(out.view(N, self.num_heads, self.kv_lora_rank))

    def _forward_mixed(self, q, kv_c_normed, k_pe, kv_b_proj, kv_cache, ctx):
        """Mixed batch for dense (non-sparse) attention."""
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty(np_ + nd, self.num_heads * self.v_head_dim,
                          dtype=q.dtype, device=q.device)

        if np_ > 0:
            q_pf = q[:np_]
            kv_c_pf = kv_c_normed[:np_]
            k_pe_pf = k_pe[:np_]

            has_context = ctx.chunked_context is not None
            kv = kv_b_proj(kv_c_pf)
            kv = kv.view(np_, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe_pf)

            output_prefill = self._run_prefill_new_tokens(
                q_pf, k, v,
                cu_seqlens_q=ctx.prefill_cu_seqlens_q,
                cu_seqlens_k=ctx.prefill_cu_seqlens_q,
                max_seqlen_q=ctx.prefill_max_seqlen_q,
                max_seqlen_k=ctx.prefill_max_seqlen_q,
                return_softmax_lse=has_context,
            )

            if has_context:
                suffix_output, suffix_lse = output_prefill
                context_output, context_lse = self._compute_prefill_context(
                    q_pf, kv_cache, kv_b_proj, ctx)

                pf_result = torch.empty(np_, self.num_heads, self.v_head_dim,
                                        dtype=q.dtype, device=q.device)
                self.merge_states(
                    output=pf_result,
                    prefix_output=context_output,
                    prefix_lse=context_lse,
                    suffix_output=suffix_output[..., :self.v_head_dim],
                    suffix_lse=suffix_lse,
                )
                out[:np_] = pf_result.reshape(np_, self.num_heads * self.v_head_dim)
            else:
                pf_out = output_prefill
                if isinstance(pf_out, tuple):
                    pf_out = pf_out[0]
                out[:np_] = pf_out.reshape(np_, self.num_heads * self.v_head_dim)

        if nd > 0:
            q_dc = q[np_:]
            cache_seqlens = ctx.decode_context_lens
            block_table = ctx.decode_block_tables

            if not self.use_fp8_kv_cache:
                q_dc = self._absorb_q_to_latent(q_dc)
                q_dc = q_dc.unsqueeze(1)
                tile_sched_meta, _ = self.get_metadata(
                    cache_seqlens, self.num_heads, num_heads_k=1,
                )
                o, _ = self.decode_op(
                    q_dc,
                    kv_cache.unsqueeze(-2),
                    block_table,
                    cache_seqlens,
                    head_dim_v=_MLA_HEAD_DIM_V,
                    tile_scheduler_metadata=tile_sched_meta,
                    softmax_scale=self.scale,
                    causal=True,
                )
                o = o.reshape(-1, o.shape[-2], o.shape[-1])
                out[np_:] = self._v_up_proj(o)
            elif self.decode_op_fp8.available:
                tile_sched_meta, num_splits = self.get_metadata_dense_fp8(
                    cache_seqlens, self.num_heads, num_heads_k=1,
                )
                o, _ = self.decode_op_fp8(
                    q_dc.unsqueeze(1), kv_cache.view(torch.uint8).unsqueeze(-2),
                    block_table, cache_seqlens,
                    head_dim_v=_MLA_HEAD_DIM_V,
                    tile_scheduler_metadata=tile_sched_meta,
                    num_splits=num_splits,
                    softmax_scale=self.scale,
                    causal=True,
                    descale_q=self._q_scale.reshape(1),
                    descale_k=self._k_scale.reshape(1),
                )
                o = o.reshape(-1, o.shape[-2], o.shape[-1])
                out[np_:] = self._v_up_proj(o)
            else:
                tile_sched_meta, _ = self.get_metadata(
                    cache_seqlens, self.num_heads, num_heads_k=1,
                    is_fp8_kvcache=True)
                o, _ = self.decode_op(
                    q_dc.unsqueeze(1), kv_cache.view(torch.uint8).unsqueeze(-2),
                    block_table, cache_seqlens,
                    head_dim_v=_MLA_HEAD_DIM_V,
                    tile_scheduler_metadata=tile_sched_meta,
                    softmax_scale=self.scale,
                    causal=True,
                    is_fp8_kvcache=True,
                )
                o = o.reshape(-1, o.shape[-2], o.shape[-1])
                out[np_:] = self._v_up_proj(o)

        return out
