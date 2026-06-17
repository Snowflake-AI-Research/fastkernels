"""Transformer2DModel for SDXL UNet (L3).

Wraps N BasicTransformerBlocks with GroupNorm + flatten/unflatten for
spatial features (B,C,H,W) -> (B,H*W,C) -> transformer -> (B,C,H,W).

When use_linear_projection=True (SDXL default), proj_in and proj_out are
Linear layers. Otherwise they are 1x1 Conv2d.

Parameter names match diffusers' Transformer2DModel:
  norm, proj_in, transformer_blocks.N, proj_out
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.group_norm import GroupNorm
from ..L1.linear import Linear
from .sdxl_transformer_block import BasicTransformerBlock


class Transformer2DModel(nn.Module):
    """Spatial transformer: GroupNorm -> flatten -> Linear -> N x BasicTransformerBlock -> Linear -> unflatten."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int = 1,
        cross_attention_dim: int | None = None,
        norm_num_groups: int = 32,
        use_linear_projection: bool = True,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection
        self.inner_dim = inner_dim

        self.norm = GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6,
        )

        self.proj_in = Linear(in_channels, inner_dim, bias=True)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ])

        self.proj_out = Linear(inner_dim, in_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # (B, C, H, W) -> (B, H*W, C)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, -1
        )

        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.proj_out(hidden_states)

        # (B, H*W, C) -> (B, C, H, W)
        hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(
            0, 3, 1, 2
        )

        output = hidden_states + residual
        return output
