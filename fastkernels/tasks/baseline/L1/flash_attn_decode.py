"""Flash attention decode kernel (with paged KV cache).

Uses the unified ``flash_attn_varlen_func`` interface from vLLM's bundled
FlashAttention build, at the version vLLM would select for this device
(FA3 on Hopper, FA4 on Blackwell, FA2 otherwise) -- see :mod:`fa_utils`.
That is also the only build whose paged-KV path accepts the engine's block
size; upstream ``flash_attn`` requires page sizes divisible by 256.
"""

import torch
import torch.nn as nn

from ....infra.fa_utils import (
    FA3_CUDA_GRAPH_MAX_NUM_SPLITS,
    FA_VERSION,
    fa3_scheduler_metadata,
    fa3_scheduler_metadata_size,
    flash_attn_varlen_func,
)


class FlashAttnDecode(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 page_size: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._cu_seqlens_q = None
        # Persistent FA3 scheduler metadata.  vLLM builds this in
        # ``FlashAttentionMetadataBuilder`` *outside* the CUDA graph.
        self._sched_buf: torch.Tensor | None = None
        self._sched_meta: torch.Tensor | None = None
        self._graph_num_splits = FA3_CUDA_GRAPH_MAX_NUM_SPLITS
        self._window_size = (-1, -1)
        self._qkv_dtype = torch.bfloat16

    def _get_cu_seqlens_q(self, n: int, device: torch.device) -> torch.Tensor:
        needed = n + 1
        if self._cu_seqlens_q is None or self._cu_seqlens_q.numel() < needed:
            self._cu_seqlens_q = torch.arange(
                needed, dtype=torch.int32, device=device,
            )
        return self._cu_seqlens_q[:needed]

    def update_scheduler_metadata(
        self,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        qkv_dtype: torch.dtype | None = None,
        window_size: tuple[int, int] = (-1, -1),
        max_seqlen_q: int = 1,
    ) -> None:
        """Recompute FA3 tile-scheduler metadata into the persistent buffer."""
        if FA_VERSION != 3:
            self._sched_meta = None
            return
        if qkv_dtype is not None:
            self._qkv_dtype = qkv_dtype
        self._window_size = window_size
        batch_size = int(cache_seqlens.shape[0])
        cu_seqlens_q = self._get_cu_seqlens_q(batch_size, cache_seqlens.device)
        meta = fa3_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self._qkv_dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=self.page_size,
            causal=True,
            window_size=window_size,
            num_splits=self._graph_num_splits,
        )
        if meta is None:
            self._sched_meta = None
            return
        n = int(meta.shape[0])
        need = max(n, fa3_scheduler_metadata_size(batch_size))
        if self._sched_buf is None or self._sched_buf.numel() < need:
            cap = max(need, fa3_scheduler_metadata_size(max(batch_size, 1024)))
            self._sched_buf = torch.zeros(
                cap, dtype=torch.int32, device=cache_seqlens.device,
            )
        self._sched_buf[:n].copy_(meta)
        self._sched_buf[n:].zero_()
        self._sched_meta = self._sched_buf[:n]

    def preallocate(self, max_batch_size: int, device: torch.device) -> None:
        """Allocate graph-stable cu_seqlens + scheduler buffers."""
        needed = max_batch_size + 1
        if self._cu_seqlens_q is None or self._cu_seqlens_q.numel() < needed:
            self._cu_seqlens_q = torch.arange(
                needed, dtype=torch.int32, device=device,
            )
        if FA_VERSION == 3:
            cap = fa3_scheduler_metadata_size(max(max_batch_size, 1024))
            if self._sched_buf is None or self._sched_buf.numel() < cap:
                self._sched_buf = torch.zeros(
                    cap, dtype=torch.int32, device=device,
                )
            self._sched_meta = self._sched_buf[:fa3_scheduler_metadata_size(1)]

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        max_seq_len = kwargs.pop("max_seq_len", None)
        block_table = kwargs.pop("block_table", None)
        softmax_scale = kwargs.pop("softmax_scale", None)
        kwargs.pop("causal", None)
        window_size = kwargs.get("window_size", self._window_size)

        n = q.shape[0]
        cu_seqlens_q = self._get_cu_seqlens_q(n, q.device)
        if max_seq_len is not None:
            max_seqlen_k = max_seq_len
        else:
            max_seqlen_k = int(cache_seqlens.max().item()) if cache_seqlens.numel() > 0 else 0

        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and FA_VERSION == 3:
            if self._sched_meta is None:
                raise RuntimeError(
                    "FA3 CUDA-graph capture requires "
                    "update_scheduler_metadata() outside the graph first"
                )
            meta = self._sched_meta
            num_splits = self._graph_num_splits
        elif FA_VERSION == 3 and cache_seqlens is not None:
            page_size = self.page_size
            if page_size is None and k_cache.dim() >= 2:
                page_size = k_cache.shape[1]
            meta = fa3_scheduler_metadata(
                batch_size=int(cache_seqlens.shape[0]),
                max_seqlen_q=1,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=self.num_heads,
                num_heads_kv=self.num_kv_heads,
                headdim=self.head_dim,
                cache_seqlens=cache_seqlens,
                qkv_dtype=q.dtype,
                cu_seqlens_q=cu_seqlens_q,
                page_size=page_size,
                causal=True,
                window_size=window_size,
                num_splits=0,
            )
            num_splits = 0
        else:
            meta = None
            num_splits = 0

        fa_kw = dict(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=block_table,
            fa_version=FA_VERSION,
            num_splits=num_splits,
        )
        if meta is not None:
            fa_kw["scheduler_metadata"] = meta
        fa_kw.update(kwargs)
        return flash_attn_varlen_func(**fa_kw)
