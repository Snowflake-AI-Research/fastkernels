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
    flash_attn_version_for_head,
)


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 page_size: int | None = None, fa_version: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.sm_scale = head_dim ** -0.5
        self.fa_version = (
            fa_version
            or flash_attn_version_for_head(head_dim)
            or FA_VERSION
        )
        self._graph_sched_meta: torch.Tensor | None = None
        self._graph_out: torch.Tensor | None = None

    def preallocate(self, max_tokens: int, device: torch.device) -> None:
        """Persistent FA4/FA3 ``out=`` so mixed prefill does not alloc."""
        if (
            self._graph_out is None
            or self._graph_out.shape[0] < max_tokens
        ):
            self._graph_out = torch.empty(
                max_tokens, self.num_heads, self.head_dim,
                dtype=torch.bfloat16, device=device,
            )

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        # vLLM's wrapper takes keyword args in a different order than the
        # standard flash_attn signature.  With a ``block_table`` the
        # kernel needs per-sequence ``seqused_k`` rather than cumulative
        # ``cu_seqlens_k``.
        fa_kw = dict(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            fa_version=self.fa_version,
        )
        if kwargs.get("block_table") is not None:
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            fa_kw["seqused_k"] = seqused_k
            # Prefill is usually eager.  EAGLE-3 draft-extend captures it
            # in a CUDA graph, so the schedule must come from a persistent
            # buffer filled outside capture (vLLM metadata-builder pattern).
            capturing = torch.cuda.is_current_stream_capturing()
            # vLLM pins num_splits=32 only for FULL decode CUDA graphs.
            # Prefill is compute-bound; 32 splits on 16k Q allocates ~8 GiB
            # of FA3 scratch and OOMs Hopper Gemma piecewise capture.
            # num_splits=1 matches vLLM's unsplit prefill / FA4 dense pin.
            num_splits = 1
            fa_kw["num_splits"] = num_splits
            if capturing:
                meta = self._graph_sched_meta
            else:
                meta = None
                if self.fa_version == 3:
                    page_size = self.page_size
                    if page_size is None and k.dim() >= 2:
                        page_size = k.shape[1]
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
                        num_splits=num_splits,
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
        nq = q.shape[0]
        if (
            "out" not in fa_kw
            and self._graph_out is not None
            and self._graph_out.shape[0] >= nq
        ):
            fa_kw["out"] = self._graph_out[:nq]
        return flash_attn_varlen_func(q, k, v, **fa_kw)
