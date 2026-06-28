"""ResnetBlock2D for SDXL UNet.

GroupNorm -> SiLU -> Conv2d -> time_emb_proj -> GroupNorm -> SiLU -> Conv2d
with residual connection (optional 1x1 conv shortcut when channels change).

Mirrors diffusers' ResnetBlock2D (default time_embedding_norm, no up/down).
Parameter names match diffusers checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.group_norm import GroupNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


class ResnetBlock2D(nn.Module):
    """Residual block with time-embedding injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: int | None = None,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        groups_out = groups_out or groups
        self.output_scale_factor = output_scale_factor

        self.norm1 = GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = Linear(temb_channels, out_channels, bias=True) if temb_channels else None

        self.norm2 = GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = SiLU()

        self.conv_shortcut = (
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else None
        )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(input_tensor)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor
