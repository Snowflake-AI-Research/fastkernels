"""V-JEPA 2 attention blocks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.vjepa2_rope import VJEPA2RotaryEmbedding


def _eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    head_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask.to(attn_weights.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


class VJEPA2RopeAttention(nn.Module):
    def __init__(self, config, hidden_size: int = 1024, num_attention_heads: int = 16):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_attention_heads}"
            )

        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.proj = Linear(hidden_size, hidden_size, bias=True)
        self.scaling = self.attention_head_size ** -0.5
        self.rope = VJEPA2RotaryEmbedding(
            crop_size=config.crop_size,
            patch_size=config.patch_size,
            frames_per_clip=config.frames_per_clip,
            tubelet_size=config.tubelet_size,
            head_dim=self.attention_head_size,
        )

    def _project(self, layer: Linear, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        return layer(hidden_states).view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self._project(self.query, hidden_states)
        key_layer = self._project(self.key, hidden_states)
        value_layer = self._project(self.value, hidden_states)

        key_layer = self.rope(key_layer, position_mask=position_mask)
        query_layer = self.rope(query_layer, position_mask=position_mask)

        if output_attentions or head_mask is not None:
            context_layer, attention_probs = _eager_attention_forward(
                query_layer, key_layer, value_layer, self.scaling, head_mask=head_mask,
            )
            context_layer = context_layer.transpose(1, 2).contiguous()
        else:
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            context_layer = context_layer.transpose(1, 2).contiguous()
            attention_probs = None

        context_layer = context_layer.reshape(batch_size, seq_length, self.all_head_size)
        context_layer = self.proj(context_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class VJEPA2PoolerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"
            )
        self.scale = self.head_dim ** -0.5
        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        values = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)

        if output_attentions or attention_mask is not None:
            attn_output, attn_weights = _eager_attention_forward(
                queries,
                keys,
                values,
                self.scale,
                head_mask=attention_mask,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_weights = None

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class VJEPA2PoolerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"
            )
        self.scale = self.head_dim ** -0.5
        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_seq_length, embed_dim = queries.shape
        kv_seq_length = keys.shape[1]

        queries = self.q_proj(queries).view(
            batch_size, q_seq_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        keys = self.k_proj(keys).view(
            batch_size, kv_seq_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        values = self.v_proj(values).view(
            batch_size, kv_seq_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)

        if output_attentions or attention_mask is not None:
            attn_output, attn_weights = _eager_attention_forward(
                queries,
                keys,
                values,
                self.scale,
                head_mask=attention_mask,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_weights = None

        attn_output = attn_output.reshape(batch_size, q_seq_length, embed_dim).contiguous()
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights
