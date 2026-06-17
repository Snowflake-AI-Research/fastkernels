"""Top-K per row selection for DSA indexer using vLLM's CUDA kernels."""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm._C  # noqa: F401  — registers torch.ops._C


class TopKPerRow(nn.Module):
    """Top-K per row selection for sparse attention indexer.

    Uses ``torch.ops._C.top_k_per_row_prefill`` and
    ``torch.ops._C.top_k_per_row_decode`` CUDA kernels for high performance.
    """

    def forward_prefill(
        self,
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Top-K within variable row boundaries.

        Args:
            logits: ``[M, max_seq_len]`` float32 logits.
            cu_seqlen_ks: ``[M]`` row start offsets (inclusive).
            cu_seqlen_ke: ``[M]`` row end offsets (exclusive).
            topk: number of indices to keep per row.

        Returns:
            ``indices``: ``[M, topk]`` int32.
        """
        M = logits.shape[0]
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
    ) -> torch.Tensor:
        """Top-K for decode rows with per-sequence length caps.

        Args:
            logits: ``[B * next_n, max_seq_len]`` float32 logits.
            seq_lens: ``[B]`` current sequence lengths.
            next_n: speculative tokens per sequence (row stride).
            topk: number of indices per row.

        Returns:
            ``indices``: ``[B * next_n, topk]`` int32.
        """
        total_rows = logits.shape[0]
        indices = torch.full(
            (total_rows, topk), -1, dtype=torch.int32, device=logits.device,
        )
        if total_rows == 0:
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
