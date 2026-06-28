"""Naive recurrent retention kernel.

Pure PyTorch reference implementation. The per-head decay is fixed
(data-independent): gamma_h = 1 - 2^(-5-h). State is maintained in float32
for numerical stability; outputs are cast back to input dtype.

Exposed as both a function (`naive_recurrent_retention`) and an
`nn.Module` (`NaiveRecurrentRetention`) so it complies with the L1
"every task must be an nn.Module" rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _retnet_decay(num_heads: int, device: torch.device) -> torch.Tensor:
    """Fixed per-head decay gamma_h = 1 - 2^(-5-h). Shape [H], dtype fp32."""
    h_idx = torch.arange(num_heads, device=device, dtype=torch.float32)
    return 1.0 - torch.pow(torch.tensor(2.0, device=device, dtype=torch.float32), -5.0 - h_idx)


def naive_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive loop-based retention recurrence.

    Args:
        q:  [B, H, T, K]
        k:  [B, H, T, K]
        v:  [B, H, T, V]
        scale: query scaling factor (default: K**-0.5)
        initial_state: [B, H, K, V] initial recurrent state
        output_final_state: whether to return final state

    Returns:
        o: [B, H, T, V]  output
        final_state: [B, H, K, V] or None
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    gamma = _retnet_decay(H, q.device)

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h = h + initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]

        h = h * gamma[None, :, None, None] + k_i.float()[..., None] * v_i.float()[..., None, :]
        o[:, :, i] = (q_i.float()[..., None] * h).sum(-2).to(v.dtype)

    final_state = h if output_final_state else None
    return o, final_state


class NaiveRecurrentRetention(nn.Module):
    """nn.Module wrapper around `naive_recurrent_retention` for L1 compliance."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return naive_recurrent_retention(
            q, k, v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
