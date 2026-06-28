"""RTDetrV2 multi-head self-attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dropout import Dropout
from ..L1.linear import Linear
from ..L1.softmax import Softmax


class RTDetrV2MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}")
        self.scaling = self.head_dim**-0.5
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self._softmax = Softmax(dim=-1)
        self._dropout = Dropout(p=dropout)

    def _reshape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, position_embeddings: torch.Tensor | None):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        batch_size, target_len, embed_dim = hidden_states.size()
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        else:
            hidden_states_original = hidden_states

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._reshape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._reshape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._reshape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            attention_mask = attention_mask.expand(batch_size, 1, *attention_mask.size())
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=attn_weights.dtype).masked_fill_(
                    attention_mask, -torch.inf
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = self._softmax(attn_weights)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = self._dropout(attn_weights)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped
