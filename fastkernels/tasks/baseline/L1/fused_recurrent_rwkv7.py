"""Fused recurrent RWKV7 — Triton-accelerated decode kernel.

Thin ``nn.Module`` wrapper around
``fla.ops.rwkv7.fused_mul_recurrent_rwkv7``. The ``_mul_`` variant
keeps the per-token L2-normalised key (``kk``) and the gate scalar
(``a``) as separate tensors and computes ``a = -kk``, ``b = kk * a``
internally — this matches our L2 ``RWKV7Attention`` call site exactly.

Tensor layout matches FLA's convention (``[B, T, H, K]`` for r/w/k/kk/a
and ``[B, T, H, V]`` for v).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.ops.rwkv7 import fused_mul_recurrent_rwkv7


class FusedRecurrentRWKV7(nn.Module):
    """Triton fused-mul-recurrent RWKV7 kernel."""

    def forward(
        self,
        r: torch.Tensor,  # [B, T, H, K]
        w: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        kk: torch.Tensor,  # [B, T, H, K]
        a: torch.Tensor,  # [B, T, H, K]
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return fused_mul_recurrent_rwkv7(
            r=r, w=w, k=k, v=v, kk=kk, a=a,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
