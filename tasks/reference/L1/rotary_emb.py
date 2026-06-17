"""Semantic PyTorch reference for rotary_emb.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    smooth = (
        (original_max_position_embeddings / wl - low_freq_factor)
        / (high_freq_factor - low_freq_factor)
        if low_freq_factor != high_freq_factor
        else torch.zeros_like(inv_freq)
    )
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        )
        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        self.register_buffer(
            "cos_sin_cache", torch.cat((freqs.cos(), freqs.sin()), dim=-1).float(),
            persistent=False,
        )

    @staticmethod
    def forward_native(positions, query, key, head_dim, cos_sin_cache):
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim].unsqueeze(1)
        sin = cos_sin[..., embed_dim:].unsqueeze(1)
        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)
        q1, q2 = q[..., :embed_dim], q[..., embed_dim:]
        k1, k2 = k[..., :embed_dim], k[..., embed_dim:]
        query_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        key_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return query_out.view(q_shape), key_out.view(k_shape)

    def forward_cuda(self, positions, query, key):
        cache = self.cos_sin_cache.to(query.dtype)
        query_out, key_out = self.forward_native(positions, query, key, self.head_dim, cache)
        query.copy_(query_out)
        key.copy_(key_out)
        return query, key

    def forward(self, positions, query, key):
        return self.forward_cuda(positions, query, key)
