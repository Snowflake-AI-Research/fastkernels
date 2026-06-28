"""Oasis VAE self-attention."""


from __future__ import annotations


# Inlined from tasks/reference/L1/dense_attention.py
from typing import Literal

import torch.nn as nn
import torch.nn.functional as F


class DenseAttention(nn.Module):
    """Dense multi-head attention with ``(batch, seq, heads, dim)`` layout."""

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn"] = "auto"):
        super().__init__()
        del backend

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask=None,
    ):
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        return out.permute(0, 2, 1, 3)


# Inlined from tasks/reference/L1/linear.py
import torch


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


# Inlined from tasks/reference/L1/oasis_rotary.py
from math import pi


def oasis_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def oasis_apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_left = t[..., :0]
    t_middle = t[..., :rot_dim]
    t_right = t[..., rot_dim:]
    t_transformed = (t_middle * freqs.cos()) + (oasis_rotate_half(t_middle) * freqs.sin())
    return torch.cat((t_left, t_transformed, t_right), dim=-1).to(dtype)


class OasisRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        return freqs.repeat_interleave(2, dim=-1)

    def forward(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        del seq_len, offset
        return self._forward_freqs(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=t.dtype)
        seq_freqs = self.forward(positions, freqs, seq_len=seq_len)
        return oasis_apply_rotary_emb(seq_freqs, t)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        colon = slice(None)
        all_freqs = []
        for index, dim in enumerate(dims):
            use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
            axis = [None] * len(dims)
            axis[index] = colon
            all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        self.register_buffer(
            "rotary_freqs",
            self.rotary.get_axial_freqs(frame_height, frame_width),
            persistent=False,
        )
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        q = oasis_apply_rotary_emb(self.rotary_freqs, q)
        k = oasis_apply_rotary_emb(self.rotary_freqs, k)

        seq_len = self.frame_height * self.frame_width
        q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        out = self.attn(q, k, v)
        out = out.reshape(bsz, seq_len, -1)
        return self.proj(out)
