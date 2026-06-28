"""RWKV7 decoder block.

Pre-norm residual pattern with LayerNorm (with bias):
  Layer 0: pre_norm -> attn_norm -> RWKV7Attention -> residual -> ffn_norm -> FFN -> residual
  Other:              attn_norm -> RWKV7Attention -> residual -> ffn_norm -> FFN -> residual

Built only from L1 ops (LayerNorm) and the L2 RWKV7 attention / FFN
modules. Forward signature mirrors FLA's ``RWKV7Block.forward`` so the
``v_first`` cross-layer carry threads through correctly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.rwkv7_attention import RWKV7Attention
from ..L2.rwkv7_ffn import RWKV7FeedForward


class RWKV7Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        if layer_idx == 0:
            self.pre_norm = LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                create_offset=config.norm_bias,
            )

        self.attn_norm = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            create_offset=config.norm_bias,
        )
        self.attn = RWKV7Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            decay_low_rank_dim=config.decay_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            a_low_rank_dim=config.a_low_rank_dim,
            v_low_rank_dim=config.v_low_rank_dim,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )

        self.ffn_norm = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            create_offset=config.norm_bias,
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, None, object | None]:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states

        h = self.attn_norm(residual)
        h, attentions, past_key_values, v_first = self.attn(
            hidden_states=h,
            v_first=v_first,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + h

        residual = hidden_states
        h = self.ffn_norm(hidden_states)
        hidden_states = residual + self.ffn(
            h, past_key_values=past_key_values, use_cache=use_cache,
        )

        return hidden_states, v_first, attentions, past_key_values
