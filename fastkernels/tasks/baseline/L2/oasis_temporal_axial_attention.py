"""Oasis temporal axial attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding


class OasisTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
        *,
        is_causal: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)

        q = q.reshape(bsz * height * width, self.heads, time, -1)
        k = k.reshape(bsz * height * width, self.heads, time, -1)
        v = v.reshape(bsz * height * width, self.heads, time, -1)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, causal=self.is_causal)
        out = out.reshape(bsz, height, width, time, self.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))
