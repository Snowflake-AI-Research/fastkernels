"""Semantic PyTorch reference for fused_recurrent_retention.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""


from __future__ import annotations


# Inlined from tasks/reference/L1/gla_recurrence.py
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
    b, h, t, k_dim = q.shape
    v_dim = v.shape[-1]
    if scale is None:
        scale = k_dim ** -0.5
    state = q.new_zeros(b, h, k_dim, v_dim, dtype=torch.float32)
    if initial_state is not None:
        state = state + initial_state.float()
    out = torch.zeros_like(v)
    for i in range(t):
        decay = gk[:, :, i].float().exp()
        state = state * decay[..., None] + k[:, :, i].float()[..., None] * v[:, :, i].float()[..., None, :]
        out[:, :, i] = ((q[:, :, i] * scale).float()[..., None] * state).sum(-2).to(v.dtype)
    return out, state if output_final_state else None


class NaiveRecurrentGLA(nn.Module):
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


class FusedRecurrentRetention(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del cu_seqlens
        heads = q.shape[2]
        h_idx = torch.arange(heads, dtype=torch.float32, device=q.device)
        gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32, device=q.device), -5.0 - h_idx)
        gk = torch.log(gamma).to(q.dtype).view(1, 1, heads, 1).expand_as(q)
        out, state = naive_recurrent_gla(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            gk.transpose(1, 2),
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return out.transpose(1, 2), state
