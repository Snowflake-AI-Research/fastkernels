"""Naive recurrent GLA (Gated Linear Attention) kernel.

Pure PyTorch reference implementation. State is maintained in float32
for numerical stability, outputs are cast back to input dtype.

Exposed as both a function (`naive_recurrent_gla`) and an `nn.Module`
(`NaiveRecurrentGLA`) so it complies with the L1 "every task must be an
nn.Module" rule while still being callable functionally from tests.

The same kernel covers Multi-Scale Retention (RetNet) when the gk tensor
is constant in time (gk[..., t, :] = log(gamma_h)), so we keep a single
recurrence op shared between GLA and RetNet at L1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive loop-based GLA recurrence.

    Args:
        q:  [B, H, T, K]
        k:  [B, H, T, K]
        v:  [B, H, T, V]
        gk: [B, H, T, K]  log-space forget gate (broadcast across T for RetNet)
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

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h = h + initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        gk_i = gk[:, :, i].float().exp()

        h = h * gk_i[..., None] + k_i.float()[..., None] * v_i.float()[..., None, :]
        o[:, :, i] = (q_i.float()[..., None] * h).sum(-2).to(v.dtype)

    final_state = h if output_final_state else None
    return o, final_state


class NaiveRecurrentGLA(nn.Module):
    """nn.Module wrapper around `naive_recurrent_gla` for L1 compliance."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return naive_recurrent_gla(
            q, k, v, gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
