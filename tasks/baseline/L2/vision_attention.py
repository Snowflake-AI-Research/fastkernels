"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from flash_attn.ops.triton.rotary import apply_rotary

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder (Qwen2-VL / Qwen2.5-VL / Qwen3-VL).

    All heads are attention heads (no GQA). Uses full (non-causal) attention.
    Supports TP: QKV is sharded, then gathered for RoPE, then re-sharded.
    """

    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        tp = _tp_size()
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.head_dim = projection_size // num_heads
        self.num_heads = num_heads // tp

        self.qkv = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, num_heads, bias=True,
        )
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)
        self.attn = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv(x)

        q_size = self.num_heads * self.head_dim
        q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = k.view(seq_len, batch_size, self.num_heads, self.head_dim)
        v = v.view(seq_len, batch_size, self.num_heads, self.head_dim)

        # Transpose to (batch, seq, heads, dim)
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk = torch.cat([q, k], dim=0)
            qk = apply_rotary(qk, rotary_pos_emb_cos, rotary_pos_emb_sin)
            q, k = qk.chunk(2, dim=0)

        # Flatten batch dim for varlen
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
        )

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)
