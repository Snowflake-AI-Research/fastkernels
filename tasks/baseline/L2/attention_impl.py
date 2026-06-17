"""vLLM-aligned Attention layer with paged KV cache.

Mirrors vLLM's ``Attention`` class (from
``vllm/model_executor/layers/attention/attention.py``):

    forward(query, key, value) -> torch.Tensor

Inputs and outputs are **flat** ``[N, num_heads * head_dim]`` tensors.
KV cache metadata is obtained from the global ``Context`` (via
``get_context()``), matching vLLM's ``get_forward_context()`` pattern.

Backend selection (flash_attn vs TRTLLM-gen) is handled at init time
via ``AttnBackendConfig``.  The engine discovers this module for KV cache
assignment through duck-typing (``hasattr(module, "k_cache")``).

TODO(tech-debt): CUDA graph capture is incompatible with chunked local
attention because the metadata remapping (cu_seqlens, block_tables) varies
per batch.  vLLM disables CUDA graphs when chunked local attention is
active.  If/when we add CUDA graph support, we need to handle this case.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context, get_attn_backend_config
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND

_TRITON_MIN_LAUNCH_GRID_SIZE_2D = 128
_TRITON_NUM_PAR_SOFTMAX_SEGMENTS = 16

try:
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as _triton_unified_attention,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode as _VllmKVQuantMode

    _TRITON_UNIFIED_AVAILABLE = True
except Exception:  # pragma: no cover - optional vLLM runtime dependency.
    _triton_unified_attention = None
    _VllmKVQuantMode = None
    _TRITON_UNIFIED_AVAILABLE = False


def _chunked_prefill_remap(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
    """Remap prefill metadata into chunked local-attention virtual batches.

    Follows vLLM's ``make_local_attention_virtual_batches`` algorithm: each
    original sequence is split into ``attention_chunk_size``-wide chunks that
    the kernel sees as independent sequences.

    Returns (cu_seqlens_q', cu_seqlens_k', max_seqlen_q', max_seqlen_k',
             block_tables').
    """
    device = cu_seqlens_q.device
    cu_q_np = cu_seqlens_q.cpu().numpy()
    cu_k_np = cu_seqlens_k.cpu().numpy()

    q_seqlens = cu_q_np[1:] - cu_q_np[:-1]
    k_seqlens = cu_k_np[1:] - cu_k_np[:-1]
    batch_size = len(q_seqlens)

    q_tokens_in_first_block = np.minimum(
        attention_chunk_size - ((k_seqlens - q_seqlens) % attention_chunk_size),
        q_seqlens,
    ).astype(np.int32)
    tokens_in_last_block = (
        attention_chunk_size + (k_seqlens % -attention_chunk_size)
    ).astype(np.int32)

    local_blocks = (
        1 + np.ceil(
            np.maximum(q_seqlens - q_tokens_in_first_block, 0) / attention_chunk_size
        ).astype(np.int32)
    )

    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = int(cu_num_blocks[-1])

    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1

    seqlens_q_local = np.repeat(
        q_seqlens - q_tokens_in_first_block, local_blocks,
    ).astype(np.int32)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attention_chunk_size * (arange - 1),
        attention_chunk_size,
    )[arange > 0]

    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    seqlens_k_local = np.full(virtual_batches, attention_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    cu_seqlens_k_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_k_local, out=cu_seqlens_k_local[1:])
    cu_seqlens_k_local[0] = 0

    max_seqlen_q = int(seqlens_q_local.max()) if virtual_batches > 0 else 0
    max_seqlen_k = int(seqlens_k_local.max()) if virtual_batches > 0 else 0

    cu_q_out = torch.from_numpy(cu_seqlens_q_local).to(device=device)
    cu_k_out = torch.from_numpy(cu_seqlens_k_local).to(device=device)

    block_tables_out = None
    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size

        k_seqstarts_absolute = np.repeat(k_seqlens, local_blocks) - (
            rarange * attention_chunk_size
            + np.repeat(tokens_in_last_block, local_blocks)
        )
        block_starts = k_seqstarts_absolute // block_size

        block_indices = (
            block_starts[:, None]
            + np.arange(pages_per_chunk, dtype=np.int32)
        )
        block_indices = block_indices.reshape(-1).clip(
            max=block_tables.shape[1] - 1,
        )
        batch_indices = np.repeat(
            np.arange(batch_size, dtype=np.int32),
            local_blocks * pages_per_chunk,
        )

        bi_torch = torch.from_numpy(batch_indices)
        bk_torch = torch.from_numpy(block_indices)
        block_tables_out = block_tables[bi_torch, bk_torch].view(
            virtual_batches, -1,
        )

    return cu_q_out, cu_k_out, max_seqlen_q, max_seqlen_k, block_tables_out


def _chunked_decode_remap(
    cache_seqlens: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Remap decode metadata so the kernel only attends within the last chunk.

    Returns (cache_seqlens', block_tables', max_context_len').
    """
    local_seqlens = torch.clamp(cache_seqlens, max=attention_chunk_size)
    max_context_len = int(local_seqlens.max().item()) if local_seqlens.numel() > 0 else 0

    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size
        chunk_start_page = (cache_seqlens - local_seqlens) // block_size
        offsets = torch.arange(pages_per_chunk, device=block_tables.device)
        page_indices = chunk_start_page.unsqueeze(1) + offsets
        page_indices = page_indices.clamp(max=block_tables.shape[1] - 1)
        block_tables = torch.gather(block_tables, 1, page_indices)

    return local_seqlens, block_tables, max_context_len


class Attention(nn.Module):

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None,
                 attention_chunk_size: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.sinks = sinks
        self.attention_chunk_size = attention_chunk_size

        # TODO(tech-debt): For chunked local attention layers the KV cache
        # could be limited to ``attention_chunk_size`` tokens per layer instead
        # of ``max_seq_len``, following vLLM's ``ChunkedLocalAttentionSpec``.
        # This is not needed for correctness but would reduce memory usage.
        self.k_cache = self.v_cache = torch.tensor([])

        attn_cfg = get_attn_backend_config()
        self._use_trtllm = attn_cfg.use_trtllm
        self._block_size = attn_cfg.block_size

        # Native FA3/TRTLLM path: sinks -> s_aux, sliding window -> window_size.
        self._fa3_sinks = sinks
        self._fa3_window_size = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )

        self._use_custom_op = False
        self._layer_name = ""
        self.register_buffer(
            "_triton_kv_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        self._decode_cu_seqlens_q: torch.Tensor | None = None
        self._triton_seq_threshold_3d = max(
            1, _TRITON_MIN_LAUNCH_GRID_SIZE_2D // self.num_kv_heads,
        )
        self._triton_softmax_segm_output: torch.Tensor | None = None
        self._triton_softmax_segm_max: torch.Tensor | None = None
        self._triton_softmax_segm_expsum: torch.Tensor | None = None

        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            from ..L1.flashinfer_prefill import TRTLLMPrefill
            from ..L1.flashinfer_decode import TRTLLMDecode
            self.prefill_op = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, head_size,
            )
        else:
            self.store_kvcache = StoreKVCache()
            from ..L1.flash_attn_prefill import FlashAttnPrefill
            from ..L1.flash_attn_decode import FlashAttnDecode
            self.prefill_op = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, head_size,
            )

        from ..L1.tree_attn_prefill import TreeAttnPrefill
        self.tree_attn_op = TreeAttnPrefill(
            self.num_heads, self.num_kv_heads, head_size,
        )

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        if self._use_trtllm:
            self.decode_op._workspace = workspace
            self.prefill_op._workspace = workspace

    def forward_impl(self, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        """Core attention logic, callable from both eager and custom-op paths."""
        ctx = get_context()
        N = query.shape[0]

        q = query.view(N, self.num_heads, self.head_size)
        k = key.view(N, self.num_kv_heads, self.head_size)
        v = value.view(N, self.num_kv_heads, self.head_size)

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)

        if getattr(ctx, "is_tree_verify", False):
            o = self.tree_attn_op(
                q, k_cache, v_cache,
                block_table_prefix=ctx.tree_block_table_prefix,
                cache_seqlens_prefix=ctx.tree_cache_seqlens_prefix,
                cu_seqlens_q_prefix=ctx.tree_cu_seqlens_q_prefix,
                max_seqlen_q_prefix=ctx.tree_max_seqlen_q_prefix,
                max_seqlen_k_prefix=ctx.tree_max_seqlen_k_prefix,
                page_table_expand=ctx.tree_page_table_expand,
                cache_seqlens_expand=ctx.tree_cache_seqlens_expand,
                cu_seqlens_q_expand=ctx.tree_cu_seqlens_q_expand,
                max_seqlen_k_expand=ctx.tree_num_verify_tokens,
                block_size=self._block_size,
                softmax_scale=self.scale,
            )
        elif ctx.is_mixed:
            if self.head_size > 256:
                can_use_triton = (
                    self._can_use_triton_unified(k_cache, ctx.prefill_block_tables)
                    and (ctx.num_decode_tokens == 0 or ctx.decode_block_tables is not None)
                )
                if can_use_triton:
                    o = self._forward_mixed_triton(q, k_cache, v_cache, ctx)
                else:
                    o = self._forward_mixed_torch(q, k_cache, v_cache, ctx)
                return o.reshape(N, self.num_heads * self.head_size)
            o = self._forward_mixed(q, k_cache, v_cache, ctx)
        else:
            if self.head_size > 256:
                if self._can_use_triton_unified(k_cache, ctx.block_tables):
                    o = self._forward_pure_triton(q, k_cache, v_cache, ctx)
                else:
                    o = self._forward_pure_torch(q, k, v, k_cache, v_cache, ctx)
                return o.reshape(N, self.num_heads * self.head_size)
            o = self._forward_pure(q, k, v, k_cache, v_cache, ctx)

        return o.reshape(N, self.num_heads * self.head_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            return torch.ops.kb_nano.unified_attention(
                query, key, value, self._layer_name,
            )
        return self.forward_impl(query, key, value)

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        if ctx.is_prefill:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k
            msq = ctx.max_seqlen_q
            msk = ctx.max_seqlen_k
            bt = ctx.block_tables

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            if bt is not None:
                return self.prefill_op(
                    q, k_cache, v_cache,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=msq, max_seqlen_k=msk,
                    softmax_scale=self.scale, causal=True,
                    block_table=bt, **fa_extra,
                )
            return self.prefill_op(
                q, k, v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                **fa_extra,
            )

        cache_seqlens = ctx.context_lens
        bt = ctx.block_tables
        max_ctx = ctx.max_context_len

        if self.attention_chunk_size is not None:
            cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                cache_seqlens, bt, self.attention_chunk_size, self._block_size,
            )

        return self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, block_table=bt,
            softmax_scale=self.scale, causal=True,
            max_seq_len=max_ctx, **fa_extra,
        )

    def _can_use_triton_unified(
        self,
        k_cache: torch.Tensor,
        block_tables: torch.Tensor | None,
    ) -> bool:
        return (
            _TRITON_UNIFIED_AVAILABLE
            and not self._use_trtllm
            and self.attention_chunk_size is None
            and k_cache.numel() > 0
            and block_tables is not None
        )

    def _get_decode_cu_seqlens_q(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        needed = num_tokens + 1
        cached = self._decode_cu_seqlens_q
        if cached is None or cached.device != device or cached.numel() < needed:
            cached = torch.arange(needed, dtype=torch.int32, device=device)
            self._decode_cu_seqlens_q = cached
        return cached[:needed]

    def _triton_kv_descale(
        self,
        num_seqs: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        return self._triton_kv_scale.expand(num_seqs, num_kv_heads)

    def _get_triton_3d_buffers(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self._triton_softmax_segm_output
        if output is None or output.device != device:
            threshold = self._triton_seq_threshold_3d
            segments = _TRITON_NUM_PAR_SOFTMAX_SEGMENTS
            head_dim_padded = 1 << (self.head_size - 1).bit_length()
            self._triton_softmax_segm_output = torch.empty(
                (threshold, self.num_heads, segments, head_dim_padded),
                dtype=torch.float32,
                device=device,
            )
            self._triton_softmax_segm_max = torch.empty(
                (threshold, self.num_heads, segments),
                dtype=torch.float32,
                device=device,
            )
            self._triton_softmax_segm_expsum = torch.empty(
                (threshold, self.num_heads, segments),
                dtype=torch.float32,
                device=device,
            )
        return (
            self._triton_softmax_segm_output,
            self._triton_softmax_segm_max,
            self._triton_softmax_segm_expsum,
        )

    def _forward_paged_triton(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = int(seqused_k.shape[0])
        kv_descale = self._triton_kv_descale(num_seqs, k_cache.shape[2])
        triton_extra = {}
        if max_seqlen_q == 1 and num_seqs <= self._triton_seq_threshold_3d:
            segm_output, segm_max, segm_expsum = self._get_triton_3d_buffers(
                q.device,
            )
            triton_extra = {
                "seq_threshold_3D": self._triton_seq_threshold_3d,
                "num_par_softmax_segments": _TRITON_NUM_PAR_SOFTMAX_SEGMENTS,
                "softmax_segm_output": segm_output,
                "softmax_segm_max": segm_max,
                "softmax_segm_expsum": segm_expsum,
            }
        _triton_unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self._fa3_window_size,
            block_table=block_tables,
            softcap=0.0,
            q_descale=None,
            k_descale=kv_descale,
            v_descale=kv_descale,
            sinks=self._fa3_sinks,
            kv_quant_mode=_VllmKVQuantMode.NONE,
            **triton_extra,
        )
        return out

    def _forward_pure_triton(self, q, k_cache, v_cache, ctx):
        if ctx.is_prefill:
            seqused_k = ctx.cu_seqlens_k[1:] - ctx.cu_seqlens_k[:-1]
            return self._forward_paged_triton(
                q,
                k_cache,
                v_cache,
                ctx.cu_seqlens_q,
                seqused_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                ctx.block_tables,
            )

        cu_q = self._get_decode_cu_seqlens_q(q.shape[0], q.device)
        return self._forward_paged_triton(
            q,
            k_cache,
            v_cache,
            cu_q,
            ctx.context_lens,
            1,
            ctx.max_context_len,
            ctx.block_tables,
        )

    def _repeat_kv_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _cache_seq(self, cache: torch.Tensor, block_table: torch.Tensor,
                   length: int) -> torch.Tensor:
        pages = (length + self._block_size - 1) // self._block_size
        block_ids = block_table[:pages].to(torch.long)
        return cache[block_ids].reshape(
            -1, self.num_kv_heads, self.head_size,
        )[:length]

    def _sdpa_one(self, q_seq: torch.Tensor, k_seq: torch.Tensor,
                  v_seq: torch.Tensor, key_offset: int = 0) -> torch.Tensor:
        q_len = q_seq.size(0)
        k_len = k_seq.size(0)
        k_seq = self._repeat_kv_for_heads(k_seq)
        v_seq = self._repeat_kv_for_heads(v_seq)
        q4 = q_seq.transpose(0, 1).unsqueeze(0)
        k4 = k_seq.transpose(0, 1).unsqueeze(0)
        v4 = v_seq.transpose(0, 1).unsqueeze(0)
        q_pos = key_offset + torch.arange(q_len, device=q_seq.device)
        k_pos = torch.arange(k_len, device=q_seq.device)
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask,
            dropout_p=0.0, scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1)

    def _prefill_torch_from_tensors(self, q, k, v, cu_q, cu_k) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = cu_q.numel() - 1
        for i in range(num_seqs):
            qs = int(cu_q[i].item())
            qe = int(cu_q[i + 1].item())
            ks = int(cu_k[i].item())
            ke = int(cu_k[i + 1].item())
            key_offset = (ke - ks) - (qe - qs)
            out[qs:qe] = self._sdpa_one(
                q[qs:qe], k[ks:ke], v[ks:ke], key_offset=key_offset,
            )
        return out

    def _prefill_torch_from_cache(self, q, k_cache, v_cache, cu_q, cu_k,
                                  block_tables) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = cu_q.numel() - 1
        for i in range(num_seqs):
            qs = int(cu_q[i].item())
            qe = int(cu_q[i + 1].item())
            k_len = int((cu_k[i + 1] - cu_k[i]).item())
            q_len = qe - qs
            k_seq = self._cache_seq(k_cache, block_tables[i], k_len)
            v_seq = self._cache_seq(v_cache, block_tables[i], k_len)
            out[qs:qe] = self._sdpa_one(
                q[qs:qe], k_seq, v_seq, key_offset=k_len - q_len,
            )
        return out

    def _decode_torch(self, q, k_cache, v_cache, cache_seqlens,
                      block_tables) -> torch.Tensor:
        out = torch.empty_like(q)
        for i in range(q.size(0)):
            k_len = int(cache_seqlens[i].item())
            k_seq = self._cache_seq(k_cache, block_tables[i], k_len)
            v_seq = self._cache_seq(v_cache, block_tables[i], k_len)
            out[i:i + 1] = self._sdpa_one(
                q[i:i + 1], k_seq, v_seq, key_offset=k_len - 1,
            )
        return out

    def _forward_pure_torch(self, q, k, v, k_cache, v_cache, ctx):
        if ctx.is_prefill:
            if ctx.block_tables is not None and k_cache.numel():
                return self._prefill_torch_from_cache(
                    q, k_cache, v_cache,
                    ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                    ctx.block_tables,
                )
            return self._prefill_torch_from_tensors(
                q, k, v, ctx.cu_seqlens_q, ctx.cu_seqlens_k,
            )
        return self._decode_torch(
            q, k_cache, v_cache, ctx.context_lens, ctx.block_tables,
        )

    def _forward_mixed(self, q, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
            msq = ctx.prefill_max_seqlen_q
            msk = ctx.prefill_max_seqlen_k
            bt = ctx.prefill_block_tables

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            pq = q[:np_].contiguous() if self._use_trtllm else q[:np_]
            out[:np_] = self.prefill_op(
                pq, k_cache, v_cache,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                block_table=bt, **fa_extra,
            )

        if nd > 0:
            cache_seqlens = ctx.decode_context_lens
            bt = ctx.decode_block_tables
            max_ctx = ctx.decode_max_context_len

            if self.attention_chunk_size is not None:
                cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                    cache_seqlens, bt,
                    self.attention_chunk_size, self._block_size,
                )

            out[np_:] = self.decode_op(
                q[np_:], k_cache, v_cache,
                cache_seqlens=cache_seqlens, block_table=bt,
                softmax_scale=self.scale, causal=True,
                max_seq_len=max_ctx, **fa_extra,
            )
        return out

    def _forward_mixed_triton(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            prefill_seqused_k = (
                ctx.prefill_cu_seqlens_k[1:] - ctx.prefill_cu_seqlens_k[:-1]
            )
            out[:np_] = self._forward_paged_triton(
                q[:np_],
                k_cache,
                v_cache,
                ctx.prefill_cu_seqlens_q,
                prefill_seqused_k,
                ctx.prefill_max_seqlen_q,
                ctx.prefill_max_seqlen_k,
                ctx.prefill_block_tables,
            )

        if nd > 0:
            cu_q = self._get_decode_cu_seqlens_q(nd, q.device)
            out[np_:] = self._forward_paged_triton(
                q[np_:],
                k_cache,
                v_cache,
                cu_q,
                ctx.decode_context_lens,
                1,
                ctx.decode_max_context_len,
                ctx.decode_block_tables,
            )
        return out

    def _forward_mixed_torch(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)
        if np_ > 0:
            out[:np_] = self._prefill_torch_from_cache(
                q[:np_],
                k_cache,
                v_cache,
                ctx.prefill_cu_seqlens_q,
                ctx.prefill_cu_seqlens_k,
                ctx.prefill_block_tables,
            )
        if nd > 0:
            out[np_:] = self._decode_torch(
                q[np_:],
                k_cache,
                v_cache,
                ctx.decode_context_lens,
                ctx.decode_block_tables,
            )
        return out
