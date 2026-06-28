"""Fused recurrent GLA — Triton-accelerated decode kernel.

Thin ``nn.Module`` wrapper around ``fla.ops.gla.fused_recurrent_gla``.
This is the SOTA decode-step path for GLA / RetNet (RetNet uses the
same kernel with a constant-in-time gk).

Tensor layout matches FLA's convention (``[B, T, H, K]``) so we don't
insert a transpose on the hot path. State is ``[N, H, K, V]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.ops.gla import fused_recurrent_gla


class FusedRecurrentGLA(nn.Module):
    """Triton fused-recurrent GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        gk: torch.Tensor | None = None,  # [B, T, H, K]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return fused_recurrent_gla(
            q=q, k=k, v=v, gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
