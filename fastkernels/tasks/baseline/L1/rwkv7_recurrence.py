"""Naive recurrent RWKV7 kernel (DPLR delta rule).

Pure PyTorch reference implementation. Implements the Diagonal Plus Low-Rank
state update:
    S_t = diag(exp(w_t)) @ S_{t-1}
        + (kk_t * a_t) @ ((-kk_t)^T @ S_{t-1})
        + k_t @ v_t^T
    o_t = r_t^T @ S_t

State is maintained in float32 for numerical stability.

Exposed as both a function (`naive_recurrent_rwkv7`) and an `nn.Module`
(`NaiveRecurrentRWKV7`) so it complies with the L1 "every task must be an
nn.Module" rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def naive_recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive loop-based RWKV7 DPLR recurrence.

    Args:
        r:  [B, H, T, K]  receptance (query)
        w:  [B, H, T, K]  log-space decay (negative values)
        k:  [B, H, T, K]  key (after k_update)
        v:  [B, H, T, V]  value
        kk: [B, H, T, K]  L2-normalized key
        a:  [B, H, T, K]  sigmoid gate for low-rank update
        scale: receptance scaling factor (default: 1.0)
        initial_state: [B, H, K, V] initial recurrent state
        output_final_state: whether to return final state

    Returns:
        o: [B, H, T, V]  output
        final_state: [B, H, K, V] or None
    """
    B, H, T, K = r.shape
    V = v.shape[-1]

    h = r.new_zeros(B, H, K, V, dtype=torch.float32)
    o = torch.zeros_like(v)

    if initial_state is not None:
        h = h + initial_state.float()

    for i in range(T):
        r_i = r[:, :, i].float() * scale
        w_i = w[:, :, i].float()
        k_i = k[:, :, i].float()
        v_i = v[:, :, i].float()
        kk_i = kk[:, :, i].float()
        a_i = a[:, :, i].float()

        act_a = -kk_i
        bb = kk_i * a_i

        # Diagonal decay + low-rank correction (both use OLD state h)
        correction = (act_a[..., None] * h).sum(-2)
        h = w_i.exp()[..., None] * h + bb[..., None] * correction[..., None, :]
        h = h + k_i[..., None] * v_i[..., None, :]

        o[:, :, i] = (h * r_i[..., None]).sum(-2).to(v.dtype)

    final_state = h if output_final_state else None
    return o, final_state


class NaiveRecurrentRWKV7(nn.Module):
    """nn.Module wrapper around `naive_recurrent_rwkv7` for L1 compliance."""

    def forward(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return naive_recurrent_rwkv7(
            r, w, k, v, kk, a,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
