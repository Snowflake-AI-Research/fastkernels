"""BitNet attention block (W1.58A8 with fused QKV projection).

GQA attention with W1.58A8 :class:`BitLinearMerged` for the fused
``qkv_proj`` and :class:`BitLinear` for ``o_proj``, NeoX-style RoPE,
and an RMSNorm sub-norm applied between the attention output and the
output projection.

Architecture (per Microsoft's ``microsoft/bitnet-b1.58-2B-4T``)::

    x ─► qkv_proj (BitLinearMerged) ─► [q; k; v]
       ─► RoPE
       ─► Attention(KV cache)
       ─► attn_sub_norm (RMSNorm)
       ─► o_proj (BitLinear)

Mirrors the SOTA reference (``vllm_repo/BitNet/gpu/model.py``) which
fuses Q/K/V into a single ``wqkv`` projection.  The HF checkpoint stores
``q_proj`` / ``k_proj`` / ``v_proj`` separately; ``packed_modules_mapping``
in :class:`BitNetForCausalLM` redirects the three tensor names into the
fused parameter using shard ids ``"q"``/``"k"``/``"v"``.

Composition: the only L1 ops invoked here are :class:`BitLinear`,
:class:`BitLinearMerged`, :class:`RMSNorm`, and the rotary embedding (an
L1 op held by reference); the attention kernel itself is the L2
:class:`Attention` (paged KV cache wrapper around flash-attn).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.bitnet_linear import BitLinear, BitLinearMerged
from ..L1.bitnet_rms_norm import BitNetRMSNorm as RMSNorm
from .attention_impl import Attention


class BitNetAttention(nn.Module):
    """W1.58A8 GQA attention with sub-norm and NeoX RoPE.

    No tensor parallelism: BitNet b1.58-2B has only 5 KV heads, which makes
    head-level TP awkward.  Weights are replicated when ``tp_size > 1``.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rotary_emb: nn.Module,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb

        q_size = num_attention_heads * head_dim
        kv_size = num_key_value_heads * head_dim
        self.q_size = q_size
        self.kv_size = kv_size

        self.qkv_proj = BitLinearMerged(
            hidden_size, [q_size, kv_size, kv_size],
            shard_id_map={"q": 0, "k": 1, "v": 2},
            bias=False,
        )
        self.o_proj = BitLinear(q_size, hidden_size, bias=False)

        self.attn_sub_norm = RMSNorm(q_size, eps=rms_norm_eps)

        self.attn = Attention(
            self.num_heads, head_dim, head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        attn_output = self.attn_sub_norm(attn_output)
        return self.o_proj(attn_output)
