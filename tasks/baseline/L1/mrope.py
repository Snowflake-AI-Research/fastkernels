"""Multi-dimensional Rotary Position Embedding (M-RoPE) for Qwen VL models.

Handles 3D position tensors (3, seq_len) representing temporal/height/width
dimensions. Each dimension's positions index into a shared cos/sin cache,
and the resulting embeddings are assembled by section into the rotary dim.

Uses a Triton kernel for multimodal prefill (3D positions differ across dims)
and a custom CUDA kernel for decode / text-only (all 3 dims identical -> standard RoPE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .csrc import _C  # noqa: F401
from . import rotary_emb as _rotary_emb_reg  # noqa: F401 — registers kb_nano_rope ops


@triton.jit
def _mrope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    num_tokens,
    n_qh: tl.constexpr, n_kh: tl.constexpr,
    hd: tl.constexpr, rd: tl.constexpr,
    pad_n_qh: tl.constexpr, pad_n_kh: tl.constexpr, pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    half_rd = rd // 2
    t_cos = cos_ptr + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin_ptr + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_half_k_offsets = (
        tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < rd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=first_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=first_k_mask)


class MRotaryEmbedding(nn.Module):
    """M-RoPE for Qwen2-VL / Qwen3-VL.

    positions can be either:
      - 1D (seq_len,) for text-only (all 3 dims identical -> standard RoPE)
      - 2D (3, seq_len) for multimodal (T/H/W positions differ)

    mrope_section: list of 3 ints [t, h, w] summing to rotary_dim // 2
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        assert sum(mrope_section) == head_dim // 2

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_position_embeddings * 4, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _apply_sgl_rope(self, positions_1d, query, key):
        """Apply standard RoPE for 1D positions (decode or text-only)."""
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        if torch.compiler.is_compiling():
            from .rotary_emb import RotaryEmbedding
            return RotaryEmbedding.forward_native(
                positions_1d,
                query.view(query.shape[0], -1),
                key.view(key.shape[0], -1),
                self.head_dim, cache,
            )
        torch.ops.kb_nano_rope.rotary_embedding(
            positions_1d,
            query.view(query.shape[0], -1),
            key.view(key.shape[0], -1),
            self.head_dim,
            cache,
            True,
        )
        return query, key

    def forward_native_2d(self, positions, query, key):
        """Pure PyTorch MRoPE for (3, seq_len) positions -- Inductor-friendly.

        Mirrors the Triton _mrope_kernel: splits q/k into first/second half,
        gathers cos/sin per T/H/W section, and applies the standard neox-style
        rotation to all head_dim elements.
        """
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)

        num_tokens = query.shape[0]
        cos_sin = cache[positions]          # (3, seq_len, head_dim)
        cos, sin = cos_sin.chunk(2, dim=-1) # each (3, seq_len, head_dim/2)

        if self.mrope_interleaved:
            cos = self._apply_interleaved(cos)
            sin = self._apply_interleaved(sin)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
        # cos, sin: (seq_len, head_dim/2)

        hd = self.head_dim
        half = hd // 2
        q_shape = query.shape
        k_shape = key.shape
        q = query.view(num_tokens, -1, hd)
        k = key.view(num_tokens, -1, hd)

        cos = cos.unsqueeze(1)  # (seq_len, 1, head_dim/2)
        sin = sin.unsqueeze(1)

        q1 = q[..., :half]
        q2 = q[..., half:]
        k1 = k[..., :half]
        k2 = k[..., half:]

        new_q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        new_k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        return new_q.view(q_shape), new_k.view(k_shape)

    def forward(self, positions, query, key):
        """Apply M-RoPE in-place.

        Args:
            positions: (seq_len,) or (3, seq_len) int64 tensor
            query: (seq_len, num_heads, head_dim)
            key: (seq_len, num_kv_heads, head_dim)
        """
        if positions.ndim == 1:
            return self._apply_sgl_rope(positions, query, key)

        if torch.compiler.is_compiling():
            return self.forward_native_2d(positions, query, key)

        # 2D M-RoPE: positions (3, seq_len) with potentially different T/H/W dims (multimodal prefill)
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)

        num_tokens = positions.shape[-1]
        cos_sin = cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        cos_3d = cos.contiguous()
        sin_3d = sin.contiguous()

        hd = self.head_dim
        q_was_2d = query.ndim == 2
        if q_was_2d:
            n_qh = query.shape[1] // hd
            n_kh = key.shape[1] // hd
        else:
            n_qh = query.shape[1]
            n_kh = key.shape[1]

        q_flat = query.reshape(num_tokens, -1).contiguous()
        k_flat = key.reshape(num_tokens, -1).contiguous()
        pad_hd = triton.next_power_of_2(hd)
        pad_n_qh = triton.next_power_of_2(n_qh)
        pad_n_kh = triton.next_power_of_2(n_kh)

        _mrope_kernel[(num_tokens,)](
            q_flat, k_flat, cos_3d, sin_3d,
            num_tokens, n_qh, n_kh, hd, hd,
            pad_n_qh, pad_n_kh, pad_hd,
            self.mrope_section[0], self.mrope_section[1], self.mrope_section[2],
            self.mrope_interleaved,
        )

        return q_flat.view_as(query), k_flat.view_as(key)

    def _apply_interleaved(self, x):
        """Reorganize from [TTT...HHH...WWW] to interleaved [THWTHW...]."""
        s = self.mrope_section
        result = x[0].clone()
        result[..., 1:s[1] * 3:3] = x[1, ..., 1:s[1] * 3:3]
        result[..., 2:s[2] * 3:3] = x[2, ..., 2:s[2] * 3:3]
        return result
