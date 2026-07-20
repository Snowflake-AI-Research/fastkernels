"""Top-K per row selection for DSA indexer using vLLM's CUDA kernels."""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm._custom_ops  # noqa: F401  — registers torch.ops._C (vLLM 0.24 stable ABI)


# vLLM's DSA decode indexer uses a radix-histogram top-k fast path for the
# common index_topk sizes; see ``vllm/model_executor/layers/sparse_attn_indexer.py:337-385``.
_RADIX_TOPK_SIZES = (512, 1024, 2048)
_RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024  # bytes (matches vLLM RADIX_TOPK_WORKSPACE_SIZE)


class TopKPerRow(nn.Module):
    """Top-K per row selection for sparse attention indexer.

    Uses ``torch.ops._C.top_k_per_row_prefill`` and
    ``torch.ops._C.top_k_per_row_decode`` CUDA kernels for high performance.
    Decode additionally dispatches to the radix ``cooperative_topk`` /
    ``persistent_topk`` kernels for ``topk in {512,1024,2048}`` (the fast path
    vLLM takes for DeepSeek-V3.2 / GLM-5.2, whose ``index_topk`` is 2048).
    """

    def __init__(self):
        super().__init__()
        self._radix_ws: dict[torch.device, torch.Tensor] = {}

    def _radix_workspace(self, device: torch.device) -> torch.Tensor:
        ws = self._radix_ws.get(device)
        if ws is None:
            ws = torch.empty(_RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8,
                             device=device)
            self._radix_ws[device] = ws
        return ws

    def forward_prefill(
        self,
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Top-K within variable row boundaries.

        Args:
            logits: ``[M, max_seq_len]`` float32 logits.
            cu_seqlen_ks: ``[M]`` row start offsets (inclusive).
            cu_seqlen_ke: ``[M]`` row end offsets (exclusive).
            topk: number of indices to keep per row.
            out: optional ``[M, topk]`` int32 buffer to write into (e.g. a
                slice of the shared ``topk_indices_buffer``). Must be a
                contiguous full-width slice. Avoids a fresh alloc + copy on
                the indexer hot path; matches vLLM writing straight into
                ``topk_indices_buffer``.

        Returns:
            ``indices``: ``[M, topk]`` int32 (``out`` if provided).
        """
        M = logits.shape[0]
        if out is not None:
            indices = out
            indices.fill_(-1)
        else:
            indices = torch.full(
                (M, topk), -1, dtype=torch.int32, device=logits.device,
            )
        if M == 0:
            return indices

        torch.ops._C.top_k_per_row_prefill(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            indices,
            M,
            logits.stride(0),
            logits.stride(1),
            topk,
        )
        return indices

    def forward_decode(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        next_n: int,
        topk: int,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Top-K for decode rows with per-sequence length caps.

        Args:
            logits: ``[B * next_n, max_seq_len]`` float32 logits.
            seq_lens: ``[B]`` current sequence lengths.
            next_n: speculative tokens per sequence (row stride).
            topk: number of indices per row.
            max_seq_len: max KV length (sizes internal radix work; falls back to
                the logits column dim).

        Returns:
            ``indices``: ``[B * next_n, topk]`` int32.
        """
        total_rows = logits.shape[0]
        indices = torch.full(
            (total_rows, topk), -1, dtype=torch.int32, device=logits.device,
        )
        if total_rows == 0:
            return indices

        # Radix fast path — matches vLLM ``sparse_attn_indexer.py:337-385``. The
        # radix kernels take per-ROW lengths (not next_n), so expand the per-seq
        # ``seq_lens`` to per-row caps identical to the generic kernel's cap
        # (sampler.cu:591: ``max(0, seq_len - next_n + j + 1)``). This is the
        # branch vLLM actually executes for GLM-5.2 (index_topk=2048).
        if logits.is_cuda and topk in _RADIX_TOPK_SIZES:
            if next_n == 1:
                row_lens = seq_lens.to(torch.int32).contiguous()
            else:
                B = seq_lens.numel()
                j = torch.arange(next_n, device=seq_lens.device, dtype=torch.int32)
                row_lens = (
                    seq_lens.to(torch.int32).view(B, 1) - next_n + 1 + j.view(1, next_n)
                ).clamp_min_(0).reshape(-1).contiguous()
            msl = int(max_seq_len) if max_seq_len is not None else logits.shape[1]
            ws = self._radix_workspace(logits.device)
            cap_major = torch.cuda.get_device_capability(logits.device)[0]
            use_cooperative = (
                total_rows <= 32
                and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
                and cap_major >= 9
            )
            if use_cooperative:
                torch.ops._C.cooperative_topk(
                    logits, row_lens, indices, ws, topk, msl)
            else:
                torch.ops._C.persistent_topk(
                    logits, row_lens, indices, ws, topk, msl)
            return indices

        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            indices,
            total_rows,
            logits.stride(0),
            logits.stride(1),
            topk,
        )
        return indices
