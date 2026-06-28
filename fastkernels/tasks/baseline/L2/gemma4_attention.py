"""Gemma4 attention with per-layer RoPE and K==V full-attention support."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from .attention_impl import Attention
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from ..L1.rms_norm import RMSNorm


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        rotary_emb: nn.Module,
        rotary_dim: int,
    ):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        self.head_dim = (
            config.head_dim if self.is_sliding else config.global_head_dim
        )
        total_kv_heads = (
            config.num_key_value_heads
            if self.is_sliding or not config.attention_k_eq_v
            else config.num_global_key_value_heads
        )

        tp = _tp_size()
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = (
            total_kv_heads // tp if total_kv_heads % tp == 0 else total_kv_heads
        )
        self.rotary_emb = rotary_emb
        self.rotary_dim = rotary_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            total_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            1.0,
            num_kv_heads=self.num_kv_heads,
            sliding_window=config.sliding_window if self.is_sliding else None,
        )

    def _apply_rope(self, positions, q, k):
        if self.rotary_dim == self.head_dim:
            return self.rotary_emb(positions, q, k)

        n = q.shape[0]
        qv = q.reshape(n, self.num_heads, self.head_dim)
        kv = k.reshape(n, self.num_kv_heads, self.head_dim)
        q_rot = qv[..., :self.rotary_dim].contiguous().view(
            n, self.num_heads * self.rotary_dim,
        )
        k_rot = kv[..., :self.rotary_dim].contiguous().view(
            n, self.num_kv_heads * self.rotary_dim,
        )
        q_rot, k_rot = self.rotary_emb(positions, q_rot, k_rot)
        q = torch.cat(
            [q_rot.reshape(n, self.num_heads, self.rotary_dim),
             qv[..., self.rotary_dim:]],
            dim=-1,
        ).reshape(n, self.num_heads * self.head_dim)
        k = torch.cat(
            [k_rot.reshape(n, self.num_kv_heads, self.rotary_dim),
             kv[..., self.rotary_dim:]],
            dim=-1,
        ).reshape(n, self.num_kv_heads * self.head_dim)
        return q, k

    def forward(self, positions, hidden_states):
        n = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(n, q_size)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(n, kv_size)
        q, k = self._apply_rope(positions, q, k)
        v = self.v_norm(v.reshape(-1, self.head_dim)).reshape(n, kv_size)

        return self.o_proj(self.attn(q, k, v))
