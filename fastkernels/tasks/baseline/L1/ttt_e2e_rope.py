"""TTT-E2E rotary embedding (interleaved / GPT-J style) L1 op.

Mirrors the JAX reference at
``test-time-training/e2e:ttt/model/attention.py:apply_rotary_emb`` exactly:

    freqs = 1.0 / (theta ** (arange(0, dim, 2) / dim))      # (D/2,)
    t     = arange(end)                                      # (T,)
    freqs = outer(t, freqs)                                  # (T, D/2)
    cos, sin = cos(freqs), sin(freqs)
    # Apply: pair x[2k], x[2k+1]  →  rotate by complex (cos + i*sin)
    out[2k]   = x[2k] * cos - x[2k+1] * sin
    out[2k+1] = x[2k] * sin + x[2k+1] * cos

Distinct from :class:`L1.rotary_emb.RotaryEmbedding` (which uses the
NeOX/half-split layout ``out[k] = x[k]*cos - x[k+D/2]*sin`` and a custom
CUDA kernel). The interleaved layout is the GPT-J / JAX-default convention.

Used by TTT-E2E SWA where exact bitwise parity with the JAX reference's
RoPE matters; the layout difference between NeOX and interleaved is *not*
weight-translatable, so we have to match the reference layout directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TTTE2ERoPE(nn.Module):
    """Interleaved rotary embedding with a pre-computed cos/sin cache.

    Args:
        head_dim: embedding dimension per head (must be even).
        max_position_embeddings: how many positions to precompute.
        rope_theta: base of the geometric progression of frequencies
            (paper: 500000 for the 8K-pretrained 125m_e2e config).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        half = head_dim // 2
        freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32)[:half] / head_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        outer = torch.outer(t, freqs)                          # (T, D/2)
        self.register_buffer("cos", outer.cos(), persistent=False)
        self.register_buffer("sin", outer.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply interleaved RoPE.

        Args:
            x: ``(B, T, H, D)`` query or key
            position_ids: ``(T,)`` int positions selecting rows of cos/sin
        """
        cos = self.cos[position_ids].to(x.dtype)               # (T, D/2)
        sin = self.sin[position_ids].to(x.dtype)
        # broadcast -> (1, T, 1, D/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        orig_dtype = x.dtype
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x0 = x_pairs[..., 0]                                   # (B, T, H, D/2)
        x1 = x_pairs[..., 1]
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        return torch.stack([o0, o1], dim=-1).flatten(-2).to(orig_dtype)
