"""Chunk GLA — Triton-accelerated chunked prefill kernel.

Thin ``nn.Module`` wrapper around ``fla.ops.gla.chunk_gla``. This is
the SOTA prefill path for GLA: the chunk algorithm processes the
sequence in fixed-size chunks (default 64), running an inter-chunk
recurrent pass in fp32 plus an intra-chunk parallel pass that exploits
GPU matmul throughput.

Tensor layout matches FLA's convention (``[B, T, H, K]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.ops.gla import chunk_gla


class ChunkGLA(nn.Module):
    """Triton chunk GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        g: torch.Tensor,  # [B, T, H, K]  log-space forget gate
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_gla(
            q=q, k=k, v=v, g=g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
