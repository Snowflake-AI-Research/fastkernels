"""FA3 cascade attention for the EAGLE-3 verify step.

Direct port of sglang's two-stage strategy
(`flashattention_backend.py`, ``use_cascade_attn=True``):

1. **Prefix pass** -- a single batched FlashAttention call where every draft
   query in sequence ``i`` attends to *all* prefix tokens of sequence ``i``.
   The KV cache is read at its native paged layout (``page_size = block_size``)
   via the existing block table for the target. ``causal=False`` is correct
   because every draft token's logical position is strictly greater than every
   prefix position, so the noncausal mask reduces to "attend to the full
   prefix".

2. **Expand pass** -- a single batched FlashAttention call where each draft
   query is its own length-1 "sequence" attending only to its tree-ancestor
   draft tokens (per the verify tree mask).

   With FA3, this is paged at ``page_size = 1`` (each "page" is a single
   token); ``page_table_expand`` is sorted so the first
   ``cache_seqlens_expand[i]`` entries are the live attended slots.

   With FA2 (which requires page_size divisible by 256 for paged KV) we
   instead gather the relevant draft K/V into a small contiguous buffer
   (size <= B*N*N tokens, e.g. <=2048) and use the non-paged FA2 varlen
   path. This is mathematically identical and a tiny one-time cost
   relative to the prefix pass.

3. **LSE merge** -- combine the two outputs exactly using the standard
   log-sum-exp merge, since the two key sets (prefix tokens and draft tokens)
   are disjoint.

This replaces the previous per-sequence Python SDPA loop, which was ~92%
of ``_target_verify`` time. With the cascade we do at most two batched
kernel launches per layer, identical to sglang.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .merge_state import merge_state

_SGL_FA3_AVAILABLE = False
_SGL_FA3_WITH_KVCACHE = None
_SGL_MERGE_STATE_V2 = None
try:
    from sgl_kernel import merge_state_v2 as _sgl_merge_state_v2
    from sgl_kernel.flash_attn import (
        flash_attn_with_kvcache as _sgl_flash_attn_with_kvcache,
    )
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] == 9:
            _SGL_FA3_AVAILABLE = True
            _SGL_FA3_WITH_KVCACHE = _sgl_flash_attn_with_kvcache
            _SGL_MERGE_STATE_V2 = _sgl_merge_state_v2
except ImportError:
    pass

_FA3_AVAILABLE = False
_FA3_VARLEN_FUNC = None
try:
    from vllm.vllm_flash_attn.flash_attn_interface import (
        FA3_AVAILABLE as _VLLM_FA3_AVAILABLE,
        flash_attn_varlen_func as _vllm_fa_varlen,
    )
    if _VLLM_FA3_AVAILABLE and torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc[0] == 9:
            _FA3_AVAILABLE = True
            _FA3_VARLEN_FUNC = _vllm_fa_varlen
except ImportError:
    pass

from flash_attn import flash_attn_varlen_func as _FA2_VARLEN_FUNC


def _sgl_fa3_paged(q, k_cache, v_cache, cu_seqlens_q, cache_seqlens,
                   max_seqlen_q, max_seqlen_k, page_table, softmax_scale):
    out, lse, *_ = _SGL_FA3_WITH_KVCACHE(
        q=q.contiguous(),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=False,
        return_softmax_lse=True,
    )
    return out, lse


def _fa3_paged(q, k_cache, v_cache, cu_seqlens_q, seqused_k,
               max_seqlen_q, max_seqlen_k, block_table, softmax_scale):
    out, lse = _FA3_VARLEN_FUNC(
        q, k_cache, v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=False,
        return_softmax_lse=True,
        fa_version=3,
    )
    return out, lse


def _fa2_paged(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k,
               max_seqlen_q, max_seqlen_k, block_table, softmax_scale):
    out, lse, _ = _FA2_VARLEN_FUNC(
        q, k_cache, v_cache,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=False,
        return_attn_probs=True,
    )
    return out, lse


def _fa2_dense(q, k_flat, v_flat, cu_seqlens_q, cu_seqlens_k,
               max_seqlen_q, max_seqlen_k, softmax_scale):
    out, lse, _ = _FA2_VARLEN_FUNC(
        q, k_flat, v_flat,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=False,
        return_attn_probs=True,
    )
    return out, lse


class TreeAttnPrefill(nn.Module):
    """Verify-step attention via cascade (two batched calls + LSE merge)."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table_prefix: torch.Tensor,
        cache_seqlens_prefix: torch.Tensor,
        cu_seqlens_q_prefix: torch.Tensor,
        max_seqlen_q_prefix: int,
        max_seqlen_k_prefix: int,
        page_table_expand: torch.Tensor,
        cache_seqlens_expand: torch.Tensor,
        cu_seqlens_q_expand: torch.Tensor,
        max_seqlen_k_expand: int,
        block_size: int,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q : [B*N, H_q, D]
            Verify queries (N = num_draft_tokens) flattened across batch.
        k_cache, v_cache : [num_blocks, block_size, H_kv, D]
            Paged KV cache (NHD layout). Prefix + draft tokens already written.
        block_table_prefix : [B, max_pages] int32
            Block-level page table for the prefix pass.
        cache_seqlens_prefix : [B] int32
            Prefix length per sequence (== ``t_committed_len[i]``).
        cu_seqlens_q_prefix : [B+1] int32 = [0, N, 2N, ..., B*N]
        max_seqlen_q_prefix : int = N
        max_seqlen_k_prefix : int = max(prefix lengths)
        page_table_expand : [B*N, N] int32
            Token-level slot indices for the expand pass. Each row i contains
            up to N slots; the first ``cache_seqlens_expand[i]`` are the draft
            tokens this query is allowed to attend to (sorted "live" first).
        cache_seqlens_expand : [B*N] int32
            Number of attended draft tokens per query.
        cu_seqlens_q_expand : [B*N+1] int32 = arange(B*N+1)
        max_seqlen_k_expand : int  (== N)
        block_size : int
            Page size of the underlying KV cache.
        """
        scale = softmax_scale if softmax_scale is not None else self.sm_scale
        H_kv = self.num_kv_heads
        D = self.head_dim

        kc_blk = k_cache.view(-1, block_size, H_kv, D)
        vc_blk = v_cache.view(-1, block_size, H_kv, D)

        if _SGL_FA3_AVAILABLE:
            o_prefix, lse_prefix = _sgl_fa3_paged(
                q, kc_blk, vc_blk,
                cu_seqlens_q=cu_seqlens_q_prefix,
                cache_seqlens=cache_seqlens_prefix,
                max_seqlen_q=max_seqlen_q_prefix,
                max_seqlen_k=max_seqlen_k_prefix,
                page_table=block_table_prefix,
                softmax_scale=scale,
            )

            kc_tok = k_cache.view(-1, 1, H_kv, D)
            vc_tok = v_cache.view(-1, 1, H_kv, D)
            o_expand, lse_expand = _sgl_fa3_paged(
                q, kc_tok, vc_tok,
                cu_seqlens_q=cu_seqlens_q_expand,
                cache_seqlens=cache_seqlens_expand,
                max_seqlen_q=1,
                max_seqlen_k=max_seqlen_k_expand,
                page_table=page_table_expand,
                softmax_scale=scale,
            )
            out, _ = _SGL_MERGE_STATE_V2(
                o_prefix,
                lse_prefix.T.contiguous(),
                o_expand,
                lse_expand.T.contiguous(),
            )
            return out

        if _FA3_AVAILABLE:
            o_prefix, lse_prefix = _fa3_paged(
                q, kc_blk, vc_blk,
                cu_seqlens_q=cu_seqlens_q_prefix,
                seqused_k=cache_seqlens_prefix,
                max_seqlen_q=max_seqlen_q_prefix,
                max_seqlen_k=max_seqlen_k_prefix,
                block_table=block_table_prefix,
                softmax_scale=scale,
            )

            kc_tok = k_cache.view(-1, 1, H_kv, D)
            vc_tok = v_cache.view(-1, 1, H_kv, D)
            o_expand, lse_expand = _fa3_paged(
                q, kc_tok, vc_tok,
                cu_seqlens_q=cu_seqlens_q_expand,
                seqused_k=cache_seqlens_expand,
                max_seqlen_q=1,
                max_seqlen_k=max_seqlen_k_expand,
                block_table=page_table_expand,
                softmax_scale=scale,
            )
        else:
            cu_seqlens_k_prefix = torch.zeros(
                cache_seqlens_prefix.shape[0] + 1,
                dtype=torch.int32, device=q.device,
            )
            cu_seqlens_k_prefix[1:] = torch.cumsum(
                cache_seqlens_prefix, dim=0, dtype=torch.int32,
            )
            o_prefix, lse_prefix = _fa2_paged(
                q, kc_blk, vc_blk,
                cu_seqlens_q=cu_seqlens_q_prefix,
                cu_seqlens_k=cu_seqlens_k_prefix,
                max_seqlen_q=max_seqlen_q_prefix,
                max_seqlen_k=max_seqlen_k_prefix,
                block_table=block_table_prefix,
                softmax_scale=scale,
            )

            BN, N = page_table_expand.shape
            row_arange = torch.arange(N, device=q.device)
            valid_mask = (
                row_arange[None, :] < cache_seqlens_expand[:, None].long()
            )                                                         # [B*N, N]
            flat_slots = page_table_expand[valid_mask].long()         # [total_k]
            kc_flat = k_cache.view(-1, H_kv, D)
            vc_flat = v_cache.view(-1, H_kv, D)
            k_gathered = kc_flat[flat_slots]                          # [total_k, H_kv, D]
            v_gathered = vc_flat[flat_slots]
            cu_seqlens_k_expand = torch.zeros(
                BN + 1, dtype=torch.int32, device=q.device,
            )
            cu_seqlens_k_expand[1:] = torch.cumsum(
                cache_seqlens_expand, dim=0, dtype=torch.int32,
            )
            o_expand, lse_expand = _fa2_dense(
                q, k_gathered, v_gathered,
                cu_seqlens_q=cu_seqlens_q_expand,
                cu_seqlens_k=cu_seqlens_k_expand,
                max_seqlen_q=1,
                max_seqlen_k=max_seqlen_k_expand,
                softmax_scale=scale,
            )

        out, _ = merge_state(
            o_prefix, lse_prefix,
            o_expand, lse_expand,
            lse_head_major=True,
        )
        return out
