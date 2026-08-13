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
        # ``qkv`` is [q | k | v] on the last dim, so q and k are already
        # adjacent: take them as one slice, make that slice contiguous once, and
        # let rotary see it as a single (2*batch, seq, heads, dim) tensor. v then
        # needs no copy at all -- reshape gives a contiguous view because
        # batch_size is 1 here.
        #
        # The previous version did .contiguous() separately on q, k and v and
        # then torch.cat([q, k]) -- four copies of ~147MB each at a full
        # encoder batch. A kernel profile against vLLM's encoder showed the two
        # engines running identical attention/GEMM/norm kernels, with our only
        # excess being 11.7ms/call in unrolled_elementwise<direct_copy> that
        # vLLM does not emit at all: 55% of a 21ms/call deficit, and the reason
        # our encoder peaked at 1.07x vLLM's memory.
        qk = qkv[..., : 2 * q_size].view(
            seq_len, batch_size, 2, self.num_heads, self.head_dim,
        )
        # -> (2, batch, seq, heads, dim), one copy
        qk = qk.permute(2, 1, 0, 3, 4).contiguous()

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            flat = qk.view(2 * batch_size, seq_len, self.num_heads,
                           self.head_dim)
            apply_rotary(flat, rotary_pos_emb_cos, rotary_pos_emb_sin,
                         inplace=True)

        q = qk[0].reshape(-1, self.num_heads, self.head_dim)
        k = qk[1].reshape(-1, self.num_heads, self.head_dim)
        # Keep v in the same (batch, seq) order as q/k. batch_size is 1 on every
        # current caller, but ordering v seq-major would silently disagree with
        # q/k if that ever changed.
        v = (qkv[..., 2 * q_size:]
             .view(seq_len, batch_size, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .reshape(-1, self.num_heads, self.head_dim))

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            # Disable split-KV. With ``num_splits=0`` (auto) FA4's CuTeDSL
            # kernel runs ``num_splits_heuristic`` and, for the few m-blocks a
            # TP-sharded encoder produces (num_heads // tp, e.g. 16 // 4 = 4)
            # at moderate seqlens, picks ``num_splits > 1``. That enables the
            # ``is_split_kv`` path in ``flash_fwd_sm100.py``, whose
            # ``n_block_first`` is typed ``None`` on one branch and ``Int32``
            # on another -- a TYPE_UNSTABLE_JOIN CuTe compile error on
            # Blackwell (SM100).  Encoder self-attention is balanced
            # (q_len == k_len) so split-KV never helps here; forcing 1 is
            # numerically identical and sidesteps the kernel bug.  The paged
            # LLM prefill path (block_table/seqused_k) keeps auto-splitting,
            # where short-q-over-long-KV chunks do benefit.
            num_splits=1,
        )

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)
