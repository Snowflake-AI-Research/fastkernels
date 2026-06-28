"""Gemma dense attention with simple KV cache (L2 composite).

Unlike the paged LlamaAttention (which uses FlashAttn varlen + paged KV),
this module uses DenseAttention for stateless or prefix-cached attention.
Used by Pi0's Gemma VLM backbone and action expert.

Supports GQA (num_kv_heads < num_heads) and RoPE.

Mirrors HuggingFace Transformers ``GemmaAttention`` / ``GemmaSdpaAttention``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: (batch, seq, num_heads, head_dim)
        k: (batch, seq, num_kv_heads, head_dim)
        cos: (batch, seq, head_dim)  or  (1, seq, head_dim)
        sin: same shape as cos
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def build_pi0_dit_attn_bias(
    prefix_len: int,
    suffix_len: int,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Blockwise attention bias matching HF ``PI0Model`` (kv_block <= q_block).

    Returns a tensor of shape ``(batch, 1, suffix_len, prefix_len + suffix_len)``
    suitable for :func:`torch.nn.functional.scaled_dot_product_attention``.
    """
    total_kv = prefix_len + suffix_len
    k_idx = torch.arange(total_kv, device=device, dtype=torch.float32).view(1, 1, total_kv)
    q_idx = torch.arange(
        prefix_len, prefix_len + suffix_len, device=device, dtype=torch.float32,
    ).view(1, suffix_len, 1)
    bk = (k_idx > prefix_len).to(torch.float32)
    bq = (q_idx > prefix_len).to(torch.float32)
    invalid = bk > bq
    neg_inf = torch.finfo(torch.float32).min
    bias = torch.where(
        invalid,
        torch.full_like(invalid, neg_inf, dtype=torch.float32),
        torch.zeros_like(invalid, dtype=torch.float32),
    )
    return bias.expand(batch, -1, -1).unsqueeze(1).contiguous()


class GemmaRotaryEmbedding(nn.Module):
    """Precompute RoPE cos/sin cache for Gemma.

    Returns (cos, sin) tensors indexed by position_ids.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 8192,
                 base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: any tensor (used only for device/dtype).
            position_ids: (batch, seq_len) integer positions.
        Returns:
            cos, sin each (batch, seq_len, head_dim).
        """
        inv_freq = self.inv_freq[None, :, None].float()
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq * pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class GemmaDenseAttention(nn.Module):
    """Multi-head attention for Gemma with dense (non-paged) KV cache.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Dimension per head.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)
        self.attn = DenseAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: (batch, seq, hidden_size)
            cos, sin: (batch, seq, head_dim) RoPE embeddings for current positions.
            attention_mask: Optional additive mask for SDPA, shape
                ``(batch, 1, q_seq, kv_seq)`` (e.g. Pi0 DiT block mask).
                When provided, ``causal`` is ignored.
            kv_cache: Optional (cached_k, cached_v) each
                      (batch, cached_len, num_kv_heads, head_dim).
            causal: Use causal (lower-triangular) masking. Ignored when
                ``attention_mask`` is not None.

        Returns:
            output: (batch, seq, hidden_size)
            new_kv_cache: (key, value) for this layer.
        """
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        new_kv = (k, v)

        if self.num_kv_groups > 1:
            k = k[:, :, :, None, :].expand(
                bsz, k.shape[1], self.num_kv_heads, self.num_kv_groups, self.head_dim,
            ).reshape(bsz, k.shape[1], self.num_heads, self.head_dim)
            v = v[:, :, :, None, :].expand(
                bsz, v.shape[1], self.num_kv_heads, self.num_kv_groups, self.head_dim,
            ).reshape(bsz, v.shape[1], self.num_heads, self.head_dim)

        scale = self.head_dim ** -0.5
        use_causal = causal and attention_mask is None
        out = self.attn(
            q, k, v,
            softmax_scale=scale,
            causal=use_causal,
            attn_mask=attention_mask,
        )

        out = out.reshape(bsz, seq_len, -1)
        return self.o_proj(out), new_kv
