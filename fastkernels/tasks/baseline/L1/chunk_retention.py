"""Chunk RetNet — Triton-accelerated chunked prefill kernel.

Thin ``nn.Module`` wrapper around ``fla.ops.retention.chunk_retention``.
Tensor layout matches FLA's convention (``[B, T, H, K]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.ops.retention import chunk_retention


class ChunkRetention(nn.Module):
    """Triton chunk RetNet kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_retention(
            q=q, k=k, v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
