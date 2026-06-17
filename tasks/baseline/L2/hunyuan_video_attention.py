"""HunyuanVideo-1.5 dual-stream joint attention (L2 composite).

Key difference from FluxAttention: RoPE is applied **only** to the video
stream Q/K *before* concatenation with the encoder stream, whereas Flux
applies RoPE to the concatenated Q/K.

Mirrors vllm-omni's ``HunyuanVideo15Attention`` in
``vllm_omni/diffusion/models/hunyuan_video/hunyuan_video_15_transformer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.t5_layer_norm import T5LayerNorm as RMSNorm
from ..L1.diffusion_rope import DiffusionRoPE
from ..L1.dense_attention import DenseAttention
from .parallel_linear import QKVParallelLinear, RowParallelLinear


class HunyuanVideo15Attention(nn.Module):
    """Dual-stream joint attention with TP optimization.

    Key difference from FluxAttention: RoPE is applied **only** to the video
    stream Q/K *before* concatenation with the encoder stream, whereas Flux
    applies RoPE to the concatenated Q/K.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int | None = None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            total_num_kv_heads=self.heads,
            bias=bias,
        )

        self.to_out = nn.ModuleList([
            RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Identity(),
        ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                total_num_kv_heads=self.heads,
                bias=added_proj_bias if added_proj_bias is not None else True,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
            )

        self.rope = DiffusionRoPE(is_neox_style=False)
        self.attn = DenseAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_heads = self.to_qkv.num_heads
        num_kv_heads = self.to_qkv.num_kv_heads

        qkv = self.to_qkv(hidden_states)
        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (num_heads, -1))
        key = key.unflatten(-1, (num_kv_heads, -1))
        value = value.unflatten(-1, (num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)

        if encoder_hidden_states is not None:
            add_num_heads = self.add_kv_proj.num_heads
            add_num_kv_heads = self.add_kv_proj.num_kv_heads

            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            add_q_size = add_num_heads * self.head_dim
            add_kv_size = add_num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (add_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (add_num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (add_num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        if attention_mask is not None:
            seq_len = query.shape[1]
            attention_mask = F.pad(attention_mask, (seq_len - attention_mask.shape[1], 0), value=True)
            attention_mask = attention_mask.bool()

        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = self.attn(query, key, value, softmax_scale=softmax_scale, causal=False)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = self.to_out[0](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
