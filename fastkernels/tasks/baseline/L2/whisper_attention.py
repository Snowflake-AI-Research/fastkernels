"""Whisper attention variants: encoder self-attention, decoder self-attention,
and decoder cross-attention.

Encoder self-attention has no KV cache and no causal mask (full bidirectional).
Decoder self-attention uses the standard paged KV cache with causal masking.
Cross-attention projects encoder K/V once during prefill, writes them to
paged KV cache blocks, and reads from cache on subsequent decode steps.
This matches vLLM's approach: cross-attention K/V live in the same paged
KV system as self-attention, with dedicated blocks per request.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from ....infra.context import get_context, get_attn_backend_config
from .parallel_linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.flash_attn_decode import FlashAttnDecode
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND


class WhisperEncoderSelfAttention(nn.Module):
    """Bidirectional self-attention for the Whisper encoder.

    No KV cache. Uses flash_attn_varlen_func with causal=False on batched
    [B, T, D] inputs.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        tp = _tp_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp
        self.head_dim = embed_dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim, self.head_dim,
            num_heads, num_heads,
            bias=True,
        )
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=True)
        self.prefill_op = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] or [T, D]
        Returns:
            [B, T, D] or [T, D]
        """
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)

        B, T, D = hidden_states.shape
        flat = hidden_states.reshape(B * T, D)
        qkv = self.qkv_proj(flat)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(B * T, self.num_heads, self.head_dim)
        k = k.view(B * T, self.num_heads, self.head_dim)
        v = v.view(B * T, self.num_heads, self.head_dim)

        cu_seqlens = torch.arange(
            0, (B + 1) * T, T, dtype=torch.int32, device=hidden_states.device,
        )
        out = self.prefill_op(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=T,
            max_seqlen_k=T,
            softmax_scale=self.scale,
            causal=False,
        )

        out = out.reshape(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out.reshape(B * T, -1))
        out = out.view(B, T, -1)

        if is_2d:
            out = out.squeeze(0)
        return out


class WhisperDecoderSelfAttention(nn.Module):
    """Causal self-attention for the Whisper decoder with paged KV cache.

    Uses the standard Attention implementation from attention_impl.py.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        tp = _tp_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp
        self.head_dim = embed_dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim, self.head_dim,
            num_heads, num_heads,
            bias=True,
        )
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=True)
        self.attn = Attention(self.num_heads, self.head_dim, self.scale,
                              num_kv_heads=self.num_heads)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [N, D] flat token embeddings (paged KV cache context)
        """
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        return self.out_proj(attn_output)


class WhisperCrossAttention(nn.Module):
    """Decoder cross-attention with paged KV cache.

    Matches vLLM's approach: encoder K/V are projected once during prefill
    via a fused kv_proj (QKVParallelLinear with total_num_heads=0), written
    to paged KV cache via slot_mapping. On subsequent decode steps, the
    attention reads from the paged cache.

    The engine discovers cross-attention layers via duck-typing on
    ``k_cache`` / ``v_cache`` attributes, same as self-attention layers.
    Cross-attention layers are distinguished by ``is_cross_attn = True``.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        tp = _tp_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp
        self.head_dim = embed_dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True)
        self.kv_proj = QKVParallelLinear(
            embed_dim, self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=num_heads,
            bias=True,
        )
        self.out_proj = RowParallelLinear(embed_dim, embed_dim, bias=True)

        self.is_cross_attn = True
        self.k_cache = self.v_cache = torch.tensor([])

        attn_cfg = get_attn_backend_config()
        self._block_size = attn_cfg.block_size

        self.store_kvcache = (
            StoreKVCacheHND(page_size=attn_cfg.block_size)
            if attn_cfg.use_trtllm
            else StoreKVCache()
        )
        self.prefill_op = FlashAttnPrefill(
            self.num_heads, self.num_heads, self.head_dim,
        )
        self.decode_op = FlashAttnDecode(
            self.num_heads, self.num_heads, self.head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [N_dec, D] decoder token states (flat)
            encoder_hidden_states: [N_enc, D] concatenated encoder outputs
                for NEW requests this step, or None if all requests are
                in decode phase (K/V already in paged cache).
        """
        ctx = get_context()

        q = self.q_proj(hidden_states)
        N_dec = q.shape[0]
        q = q.view(N_dec, self.num_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        if encoder_hidden_states is not None:
            kv = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            N_enc = encoder_hidden_states.shape[0]
            k = k.view(N_enc, self.num_heads, self.head_dim)
            v = v.view(N_enc, self.num_heads, self.head_dim)

            if k_cache.numel() and v_cache.numel() and ctx.cross_slot_mapping is not None:
                self.store_kvcache(k, v, k_cache, v_cache, ctx.cross_slot_mapping)

        if not k_cache.numel():
            return self.out_proj(torch.zeros_like(hidden_states))

        if ctx.is_prefill or ctx.is_mixed:
            if ctx.cross_cu_seqlens_q is None:
                return self.out_proj(torch.zeros_like(hidden_states))
            return self._forward_prefill(q, k_cache, v_cache, ctx)
        else:
            return self._forward_decode(q, k_cache, v_cache, ctx)

    def _forward_prefill(self, q, k_cache, v_cache, ctx):
        """Prefill: Q attends to encoder K/V in paged cache (non-causal)."""
        cu_q = ctx.cross_cu_seqlens_q
        cu_k = ctx.cross_cu_seqlens_k
        bt = ctx.cross_block_tables

        out = self.prefill_op(
            q, k_cache, v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=ctx.cross_max_seqlen_q,
            max_seqlen_k=ctx.cross_max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            block_table=bt,
        )
        return self.out_proj(out.view(q.shape[0], -1))

    def _forward_decode(self, q, k_cache, v_cache, ctx):
        """Decode: each decoder token attends to full encoder KV in cache."""
        out = self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=ctx.cross_context_lens,
            block_table=ctx.cross_block_tables,
            softmax_scale=self.scale,
            causal=False,
            max_seq_len=ctx.cross_max_context_len,
        )
        return self.out_proj(out.view(q.shape[0], -1))
