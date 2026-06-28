"""Chunk RWKV7 — Triton-accelerated chunked prefill kernel.

Thin ``nn.Module`` wrapper around ``fla.ops.rwkv7.chunk_rwkv7``. RWKV7
needs an explicit ``a = -kk`` and ``b = kk * a`` decomposition for the
DPLR-style update; the kernel takes those as separate args.

Tensor layout matches FLA's convention (``[B, T, H, K]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.ops.rwkv7 import chunk_rwkv7


class ChunkRWKV7(nn.Module):
    """Triton chunk RWKV7 kernel."""

    def forward(
        self,
        r: torch.Tensor,  # [B, T, H, K]
        w: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        a: torch.Tensor,  # [B, T, H, K]  (== -kk)
        b: torch.Tensor,  # [B, T, H, K]  (== kk * gate_a)
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_rwkv7(
            r=r, w=w, k=k, v=v, a=a, b=b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            safe_gate=True,
            chunk_size=chunk_size,
        )
