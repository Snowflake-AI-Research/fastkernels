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
        # ``FlashAttentionMetadataBuilder`` *outside* the CUDA graph and
        # copies it into a preallocated buffer; calling
        # ``get_scheduler_metadata`` from inside the captured region
        # records a kernel shape that IMA's on replay with real lengths
        # (same class of bug as the DSA indexer schedule).
        self._sched_buf: torch.Tensor | None = None
        self._sched_meta: torch.Tensor | None = None
        # Optional per-step buffer for multi-decode CUDA graphs (EAGLE-3
        # draft chain).  When set, capture records this pointer instead
        # of ``_sched_meta``.
        self._graph_sched_meta: torch.Tensor | None = None
        self._graph_num_splits = FA3_CUDA_GRAPH_MAX_NUM_SPLITS
        # Persistent decode output.  vLLM passes ``out=`` into FA3 so the
        # kernel does not allocate a process-wide workspace that the last
        # CUDA-graph capture then shrinks (Jamba multi-bucket IMA).
        self._graph_out: torch.Tensor | None = None
        # Set for the duration of CUDA-graph capture/warmup so eager
        # warmups use the same num_splits=32 workspace as the graph.
        self._force_graph_splits: bool = False
        self._window_size = (-1, -1)
        self._qkv_dtype = torch.bfloat16
        self._graph_q: torch.Tensor | None = None
        self._graph_seqlens: torch.Tensor | None = None
        self._graph_block_table: torch.Tensor | None = None
        self._sched_meta_batch: int = 0
        # Eager mixed decode pads to this max-B so FA3's process-wide
        # split-KV scratch cannot shrink below the largest captured graph.
        self._pad_q: torch.Tensor | None = None
        self._pad_seqlens: torch.Tensor | None = None
        self._pad_bt: torch.Tensor | None = None

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
        """Recompute FA3 tile-scheduler metadata into the persistent buffer.

        Must run OUTSIDE CUDA graph capture/replay, matching vLLM's
        metadata builder.  ``cache_seqlens`` is the [B] decode lengths
        including the padded tail the captured graph covers.
        """
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
        # Zero the tail so leftover tiles from a larger batch cannot
        # overwrite the output (vLLM ``FlashAttentionMetadataBuilder``).
        self._sched_buf[n:].zero_()
        # Pass exactly ``[:n]`` -- FA3 checks ``shape == (metadata_size)``.
        # For a fixed batch + num_splits this n is stable across capture
        # (dummy len=1) and replay (real lengths).
        self._sched_meta = self._sched_buf[:n]
        self._sched_meta_batch = batch_size

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
            if (
                self._graph_out is None
                or self._graph_out.shape[0] < max_batch_size
            ):
                self._graph_out = torch.empty(
                    max_batch_size, self.num_heads, self.head_dim,
                    dtype=self._qkv_dtype, device=device,
                )

    def _pad_eager_to_graph_batch(self, q, cache_seqlens, block_table):
        """Pad an eager FA3 decode to the captured max batch.

        Multi-bucket capture ends at B=1; a later smaller ``num_splits=0``
        call shrinks FA3's process-wide scratch and the large graph IMAs.
        Running every eager decode at ``_graph_out``'s batch keeps that
        scratch at the size capture recorded.  B200 / FA4 is unchanged
        (this is only reached for FA3).
        """
        target = int(self._graph_out.shape[0])
        n = q.shape[0]
        if (
            self._pad_q is None
            or self._pad_q.shape[0] < target
            or self._pad_q.shape[1:] != q.shape[1:]
            or self._pad_q.dtype != q.dtype
            or self._pad_q.device != q.device
        ):
            self._pad_q = torch.zeros(
                target, *q.shape[1:], dtype=q.dtype, device=q.device,
            )
            self._pad_seqlens = torch.zeros(
                target, dtype=torch.int32, device=q.device,
            )
        self._pad_q[:n].copy_(q)
        self._pad_q[n:target].zero_()
        self._pad_seqlens[:n].copy_(cache_seqlens)
        self._pad_seqlens[n:target].zero_()
        bt = block_table
        if bt is not None:
            width = int(bt.shape[1])
            if (
                self._pad_bt is None
                or self._pad_bt.shape[0] < target
                or self._pad_bt.shape[1] < width
                or self._pad_bt.device != bt.device
            ):
                self._pad_bt = torch.zeros(
                    target, width, dtype=bt.dtype, device=bt.device,
                )
            self._pad_bt[:n, :width].copy_(bt)
            self._pad_bt[n:target].zero_()
            bt = self._pad_bt[:target, :width]
        return self._pad_q[:target], self._pad_seqlens[:target], bt

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        max_seq_len = kwargs.pop("max_seq_len", None)
        block_table = kwargs.pop("block_table", None)
        softmax_scale = kwargs.pop("softmax_scale", None)
        kwargs.pop("causal", None)
        window_size = kwargs.get("window_size", self._window_size)

        orig_n = q.shape[0]
        n = orig_n
        capturing = torch.cuda.is_current_stream_capturing()
        # Pad only post-capture eager decode.  Capture warmup sets
        # ``_force_graph_splits`` so every bucket records num_splits=32;
        # padding that warmup to max-B rewrites ``_sched_meta_batch``
        # while the graph still sees native B, and FA3 then rejects the
        # metadata (or IMAs on replay).  B200 / FA4 never enters here.
        if (
            FA_VERSION == 3
            and not capturing
            and not self._force_graph_splits
            and self._graph_out is not None
            and cache_seqlens is not None
            and n < int(self._graph_out.shape[0])
        ):
            q, cache_seqlens, block_table = self._pad_eager_to_graph_batch(
                q, cache_seqlens, block_table,
            )
            n = q.shape[0]
            self.update_scheduler_metadata(
                cache_seqlens,
                max_seq_len if max_seq_len is not None else int(cache_seqlens.max().item() or 0),
                qkv_dtype=q.dtype,
                window_size=window_size,
            )
            self._force_graph_splits = True
        cu_seqlens_q = self._get_cu_seqlens_q(n, q.device)
        if max_seq_len is not None:
            max_seqlen_k = max_seq_len
        else:
            max_seqlen_k = int(cache_seqlens.max().item()) if cache_seqlens.numel() > 0 else 0

        capturing = torch.cuda.is_current_stream_capturing()
        # Capture, pin, and padded eager decode all stay on num_splits=32
        # so FA3's process-wide split-KV scratch cannot shrink below the
        # largest captured graph.  B200 / FA4 never sets _force_graph_splits.
        use_graph_splits = capturing or self._force_graph_splits
        if use_graph_splits and FA_VERSION == 3:
            meta = self._sched_meta
            if meta is None or int(getattr(self, "_sched_meta_batch", -1)) != n:
                if cache_seqlens is not None and not capturing:
                    self.update_scheduler_metadata(
                        cache_seqlens,
                        max_seqlen_k,
                        qkv_dtype=q.dtype,
                        window_size=window_size,
                    )
                    meta = self._sched_meta
            if meta is None or int(getattr(self, "_sched_meta_batch", -1)) != n:
                if capturing:
                    raise RuntimeError(
                        "FA3 CUDA-graph capture requires "
                        "update_scheduler_metadata() outside the graph first "
                        f"(q_batch={n}, sched_batch="
                        f"{getattr(self, '_sched_meta_batch', None)}, "
                        f"meta={'None' if meta is None else tuple(meta.shape)})"
                    )
                use_graph_splits = False
        if use_graph_splits:
            meta = self._sched_meta
            if FA_VERSION == 3 and meta is None:
                raise RuntimeError(
                    "FA3 CUDA-graph capture requires "
                    "update_scheduler_metadata() outside the graph first "
                    f"(q_batch={n}, sched_batch="
                    f"{getattr(self, '_sched_meta_batch', None)})"
                )
            num_splits = self._graph_num_splits
        else:
            page_size = self.page_size
            if page_size is None and k_cache.dim() >= 2:
                page_size = k_cache.shape[1]
            meta = fa3_scheduler_metadata(
                batch_size=int(cache_seqlens.shape[0]) if cache_seqlens is not None else n,
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
            ) if cache_seqlens is not None else None
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
        if self._graph_out is not None and self._graph_out.shape[0] >= n:
            fa_kw["out"] = self._graph_out[:n]
        fa_kw.update(kwargs)
        out = flash_attn_varlen_func(**fa_kw)
        return out[:orig_n]
