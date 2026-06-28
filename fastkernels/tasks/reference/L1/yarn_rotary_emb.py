"""Semantic PyTorch reference for yarn_rotary_emb.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate=False):
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_, max_, dim, dtype):
    if min_ == max_:
        max_ += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - min_) / (max_ - min_)
    return torch.clamp(linear_func, 0, 1)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _apply_rope(positions, query, key, head_dim, cache, is_neox_style):
    cos_sin = cache[positions]
    half = cos_sin.shape[-1] // 2
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)
    q_shape, k_shape = query.shape, key.shape
    q = query.view(q_shape[0], -1, head_dim)
    k = key.view(k_shape[0], -1, head_dim)
    if is_neox_style:
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    else:
        q_pair = q.view(*q.shape[:-1], half, 2)
        k_pair = k.view(*k.shape[:-1], half, 2)
        q0, q1 = q_pair[..., 0], q_pair[..., 1]
        k0, k1 = k_pair[..., 0], k_pair[..., 1]
        q_out = torch.stack([q0 * cos - q1 * sin, q1 * cos + q0 * sin], dim=-1).flatten(-2)
        k_out = torch.stack([k0 * cos - k1 * sin, k1 * cos + k0 * sin], dim=-1).flatten(-2)
    return q_out.view(q_shape), k_out.view(k_shape)


class YaRNRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        pos_freqs = rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, head_dim, rope_theta,
            original_max_position_embeddings, truncate,
        )
        inv_freq_mask = 1 - _yarn_linear_ramp_mask(low, high, head_dim // 2, dtype=torch.float)
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        mscale = yarn_get_mscale(scaling_factor)
        t = torch.arange(int(max_position_embeddings * scaling_factor), dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos() * mscale, freqs.sin() * mscale), dim=-1).float(),
            persistent=False,
        )

    def forward(self, positions, query, key):
        q, k = _apply_rope(
            positions, query, key, self.head_dim,
            self.cos_sin_cache.to(query.dtype), True,
        )
        query.copy_(q)
        key.copy_(k)
        return query, key


class YarnRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        is_neox_style: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        softmax_mscale = (
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
            * attn_factor
        )
        self.softmax_mscale = softmax_mscale
        pos_freqs = rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, head_dim, rope_theta, max_position_embeddings,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, head_dim // 2, dtype=torch.float)
        ) * extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        t = torch.arange(int(max_position_embeddings * scaling_factor), dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos() * softmax_mscale, freqs.sin() * softmax_mscale), dim=-1).float(),
            persistent=False,
        )

    def forward(self, positions, query, key):
        q, k = _apply_rope(
            positions, query, key, self.head_dim,
            self.cos_sin_cache.to(query.dtype), self.is_neox_style,
        )
        query.copy_(q)
        key.copy_(k)
        return query, key
