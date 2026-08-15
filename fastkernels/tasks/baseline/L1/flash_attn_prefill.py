"""Flash attention prefill kernel (variable-length sequences).

Routes through vLLM's bundled FlashAttention build at the version vLLM
itself would select for this device (FA3 on Hopper, FA4 on Blackwell,
FA2 otherwise) -- see :mod:`fa_utils`.
"""

import torch
import torch.nn as nn

from ....infra.fa_utils import (
    FA_VERSION,
    fa3_scheduler_metadata,
    flash_attn_varlen_func,
)


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        # vLLM's wrapper takes keyword args in a different order than the
        # standard flash_attn signature.  With a ``block_table`` the
        # kernel needs per-sequence ``seqused_k`` rather than cumulative
        # ``cu_seqlens_k``.
        fa_kw = dict(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            fa_version=FA_VERSION,
        )
        if kwargs.get("block_table") is not None:
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            fa_kw["seqused_k"] = seqused_k
            # Hopper FA3 paged prefill IMA's on long-context (16K+ KV)
            # even with scheduler metadata and num_splits=1. vLLM's FA2
            # build still accepts our page size (multiple of 16).
            if FA_VERSION == 3:
                fa_kw["fa_version"] = 2
            else:
                fa_kw["num_splits"] = 1
                page_size = k.shape[1] if k.dim() >= 2 else None
                if not torch.cuda.is_current_stream_capturing():
                    meta = fa3_scheduler_metadata(
                        batch_size=int(seqused_k.shape[0]),
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        num_heads_q=self.num_heads,
                        num_heads_kv=self.num_kv_heads,
                        headdim=self.head_dim,
                        cache_seqlens=seqused_k,
                        qkv_dtype=q.dtype,
                        cu_seqlens_q=cu_seqlens_q,
                        page_size=page_size,
                        causal=kwargs.get("causal", True),
                        window_size=kwargs.get("window_size", (-1, -1)),
                        num_splits=1,
                    )
                    if meta is not None:
                        fa_kw["scheduler_metadata"] = meta
        else:
            fa_kw["cu_seqlens_k"] = cu_seqlens_k
            # Dense prefill is compute-bound, so KV-splitting buys nothing, but
            # the FA4 (SM100 CuTe) auto heuristic still picks the split-KV kernel
            # for mid-size seqlens -- and that variant fails to compile in this
            # vLLM build (TYPE_UNSTABLE_JOIN on ``n_block_first``).  Pin
            # ``num_splits=1`` so the unsplit kernel is used.
            fa_kw["num_splits"] = 1
        fa_kw.update(kwargs)
        return flash_attn_varlen_func(q, k, v, **fa_kw)
