"""HunyuanVideo-1.5 token refiner blocks (L3 blocks).

Text conditioning encoder that refines encoder hidden states before the
main DiT transformer blocks.

Mirrors the ``HunyuanVideo15IndividualTokenRefinerBlock``,
``HunyuanVideo15IndividualTokenRefiner``, and ``HunyuanVideo15TokenRefiner``
classes in vllm-omni's ``hunyuan_video_15_transformer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L2.hunyuan_video_embeddings import HunyuanVideo15AdaNorm
from ..L2.hunyuan_video_token_refiner import (
    TokenRefinerFeedForward,
    TokenRefinerSelfAttention,
)
from ..L2.timestep_embedding import CombinedTimestepTextProjEmbeddings


class HunyuanVideo15IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = LayerNorm(hidden_size, eps=1e-6)

        self.attn = TokenRefinerSelfAttention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = LayerNorm(hidden_size, eps=1e-6)

        self.ff = TokenRefinerFeedForward(
            hidden_size, mult=mlp_width_ratio, dropout=mlp_drop_rate,
        )

        self.norm_out = HunyuanVideo15AdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideo15IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideo15IndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideo15TokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideo15IndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states
