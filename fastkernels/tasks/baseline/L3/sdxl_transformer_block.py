"""BasicTransformerBlock for SDXL UNet spatial transformers (L3).

Pre-norm transformer block with self-attention, cross-attention, and GEGLU FFN:
  LayerNorm -> SDXLAttention (self) -> residual
  LayerNorm -> SDXLAttention (cross) -> residual
  LayerNorm -> FeedForward (GEGLU) -> residual

Parameter names match diffusers' BasicTransformerBlock:
  norm1, attn1, norm2, attn2, norm3, ff
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.sdxl_attention import SDXLAttention
from ..L2.sdxl_feedforward import FeedForward


class BasicTransformerBlock(nn.Module):
    """Pre-norm transformer block with self-attn, cross-attn, and GEGLU FFN."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int | None = None,
    ):
        super().__init__()
        # promote_fp32=False: diffusers' BasicTransformerBlock uses a plain
        # nn.LayerNorm, and ATen's layer_norm already accumulates the reduction
        # in fp32 for a bf16 input, so the promotion was a parity deviation and
        # 2 extra cast kernels at each of the 3 sites per block. It also broke
        # the compiled UNet outright: the promote path caches w.float() on the
        # module, that cast runs inside the cudagraph-managed subgraph, and the
        # next invocation overwrites the pool the cached tensor points into --
        # "accessing tensor output of CUDAGraphs that has been overwritten by a
        # subsequent run", raised on the first warmup call. See L1/layer_norm.
        self.norm1 = LayerNorm(dim, promote_fp32=False)
        self.attn1 = SDXLAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        self.norm2 = LayerNorm(dim, promote_fp32=False)
        self.attn2 = SDXLAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
        )

        self.norm3 = LayerNorm(dim, promote_fp32=False)
        self.ff = FeedForward(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        norm_hidden = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden)
        hidden_states = hidden_states + attn_output

        # Cross-attention
        norm_hidden = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # FFN
        norm_hidden = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + ff_output

        return hidden_states
