"""Jamba's multi-head attention block (no RoPE, no QK-norm).

Mirrors :class:`L2.attention.LlamaAttention` exactly, sans RoPE: fused
``QKVParallelLinear`` projection -> the project's
``L2.attention_impl.Attention`` class (which handles paged-KV cache
storage and TRTLLM/FA3 dispatch) -> ``RowParallelLinear`` output
projection.

This is the *portability* point of fastkernels: the L2 task interface is
identical to vLLM's ``vllm.model_executor.models.jamba.JambaAttentionDecoderLayer``'s
attention sub-block, so a kernel optimised here drops into vLLM with
no call-site change.

Forward signature matches LlamaAttention:

    forward(positions, hidden_states) -> [N, hidden]

where ``hidden_states`` is flat varlen ``[total_tokens, hidden]`` (the
project convention; the engine packs left-padded ``[B, T]`` prompts
into this layout via ``cu_seqlens_q``).  ``positions`` is flat
``[total_tokens]`` int64 -- unused by Jamba's mixers (no RoPE; Mamba
carries position via recurrence) but threaded through for signature
parity.

Per-step paged-KV state (``slot_mapping`` / ``block_tables`` /
``context_lens`` / ``cu_seqlens_q`` / ``cu_seqlens_k``) is read by
the inner ``Attention`` from the global ``Context``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention_impl import Attention
from .parallel_linear import QKVParallelLinear, RowParallelLinear


class JambaAttention(nn.Module):
    """Fused-QKV multi-head attention, mirroring :class:`LlamaAttention`."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx

        # Fused QKV projection -- one matmul, vLLM-compatible weight shards
        # (``.weight_loader(param, tensor, shard_id)`` with shard_id in
        # {"q", "k", "v"}; the L4 weight loader translates HF's separate
        # q_proj / k_proj / v_proj weights to these three shards).
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_attention_heads,
            total_num_kv_heads=num_key_value_heads,
            bias=False,
        )
        # Output projection: row-parallel matches Llama / vLLM (no-op at TP=1).
        self.o_proj = RowParallelLinear(
            num_attention_heads * self.head_dim, hidden_size, bias=False,
        )
        # The project's standard Attention class -- handles paged-KV
        # store, FA3/TRTLLM dispatch, prefill/decode/mixed switching,
        # and torch.compile custom-op registration.  The engine binds
        # ``self.attn.k_cache`` / ``self.attn.v_cache`` to the global
        # paged cache slice for this layer, and ``auto_register_no_compile_layers``
        # picks this up by class name during engine init.
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

        # Convenient sizes for the qkv split.  vLLM also stores these on
        # the layer for the same reason.
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    def forward(
        self,
        positions: torch.Tensor | None,    # unused (no RoPE); kept for
                                            # mixer-uniform signature.
        hidden_states: torch.Tensor,        # [N, hidden] flat varlen
    ) -> torch.Tensor:
        """Fused QKV -> Attention -> o_proj.  Mirrors LlamaAttention."""
        del positions  # explicit -- Jamba attention is position-free
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
