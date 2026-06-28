"""Qwen3-Next full attention with per-head QK-norm, partial RoPE, output gating, KV cache (L2).

GQA attention: 16 query heads, 2 KV heads, head_dim=256.
Q projection outputs 2x: [Q, gate] interleaved per head.
Partial RoPE (25% of head_dim = 64 dims rotated).
Output: attn_output * sigmoid(gate).

KV cache is stored in the engine's paged state manager so Qwen3-Next can
run batched prefill/decode instead of one Python call per sequence.

Uses the existing flash-attention prefill/decode wrappers, ``GemmaRMSNorm``,
``StoreKVCache``, and the canonical TP linears in ``parallel_linear``.

Weight names match HuggingFace checkpoint:
  self_attn.q_proj.weight   [2 * num_heads * head_dim, hidden_size]  (Q + gate)
  self_attn.k_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.v_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.o_proj.weight   [hidden_size, num_heads * head_dim]
  self_attn.q_norm.weight   [head_dim]
  self_attn.k_norm.weight   [head_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.context import get_context
from ....infra.tp import _tp_size
from ..L1.flash_attn_decode import FlashAttnDecode
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L1.store_kvcache import StoreKVCache
from .parallel_linear import QKVParallelLinear, RowParallelLinear


@triton.jit
def _gate_mul_inplace_kernel(
    out_ptr,
    gate_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    out = tl.load(out_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offsets, out * gate, mask=mask)


def _gate_mul_inplace(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block = 1024
    _gate_mul_inplace_kernel[(triton.cdiv(n_elements, block),)](
        out,
        gate,
        n_elements,
        BLOCK=block,
    )
    return out


class Qwen3NextAttention(nn.Module):
    """Full attention with per-head QK-norm, partial RoPE, output gating, and KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp if num_key_value_heads % tp == 0 else num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # QKV projection: Q outputs 2x heads (Q + gate)
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads * 2,  # doubled for output gate
            num_key_value_heads,
        )

        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
        )

        # Per-head QK norms (GemmaRMSNorm)
        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)

        self.store_kvcache = StoreKVCache()
        self.flash_attn_prefill = FlashAttnPrefill(
            self.num_heads, self.num_kv_heads, self.head_dim,
        )
        self.flash_attn_decode = FlashAttnDecode(
            self.num_heads, self.num_kv_heads, self.head_dim,
        )

    def forward(self, hidden_states, rotary_emb=None, positions=None,
                state_manager=None):
        md = get_context().kda_metadata
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        N = x.shape[0]

        qkv = self.qkv_proj(x)
        q_gate_size = self.num_heads * 2 * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

        # Split Q and gate
        q_gate = q_gate.view(N, self.num_heads, 2 * self.head_dim)
        q = q_gate[:, :, :self.head_dim].contiguous()
        gate = q_gate[:, :, self.head_dim:].contiguous()

        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # Per-head QK-norm (applied before RoPE)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(N, self.num_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)

        # Partial RoPE (only rotates first rotary_dim dimensions)
        if rotary_emb is not None and positions is not None:
            pos_flat = positions.reshape(-1) if positions.dim() > 1 else positions
            rotary_dim = rotary_emb.head_dim
            q_rot, q_pass = q[..., :rotary_dim].contiguous(), q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim].contiguous(), k[..., rotary_dim:]
            q_rot, k_rot = rotary_emb(pos_flat, q_rot, k_rot)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]
        self.store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)

        out = torch.empty(
            N,
            self.num_heads,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills

        if nd > 0:
            out[:ndt] = self.flash_attn_decode(
                q[:ndt],
                k_cache,
                v_cache,
                cache_seqlens=md.seq_lens[:nd].to(torch.int32),
                block_table=md.block_tables[:nd],
                softmax_scale=self.scaling,
                causal=True,
                max_seq_len=md.max_seq_len,
            )

        if np_ > 0:
            cu_pf = (md.query_start_loc[nd:] - md.query_start_loc[nd]).to(
                torch.int32,
            )
            seqs_k = md.seq_lens[nd:]
            cu_k_pf = torch.zeros(np_ + 1, dtype=torch.int32, device=q.device)
            cu_k_pf[1:] = torch.cumsum(seqs_k.to(torch.int32), dim=0)
            out[ndt:] = self.flash_attn_prefill(
                q[ndt:],
                k_cache,
                v_cache,
                cu_seqlens_q=cu_pf,
                cu_seqlens_k=cu_k_pf,
                max_seqlen_q=md.max_query_len,
                max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                block_table=md.block_tables[nd:],
            )

        # Output gating: o * sigmoid(gate). The Triton path is faster in
        # the captured decode graph; PyTorch's vectorized path is better for
        # large prefill chunks.
        if np_ == 0:
            o = _gate_mul_inplace(out, gate)
        else:
            o = out * torch.sigmoid(gate)

        # Output projection
        o = o.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(o)
