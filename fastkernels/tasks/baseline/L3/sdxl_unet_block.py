"""UNet down/mid/up blocks for SDXL (L3).

Contains:
  - DownBlock2D: ResnetBlock2D x N + Downsample2D (no attention)
  - CrossAttnDownBlock2D: (ResnetBlock2D + Transformer2DModel) x N + Downsample2D
  - UNetMidBlock2DCrossAttn: ResnetBlock2D + Transformer2DModel + ResnetBlock2D
  - CrossAttnUpBlock2D: (ResnetBlock2D + Transformer2DModel) x N + Upsample2D
  - UpBlock2D: ResnetBlock2D x N + Upsample2D (no attention)

Parameter names match diffusers checkpoint hierarchy:
  resnets.N, attentions.N, downsamplers.0, upsamplers.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.sdxl_downsample import Downsample2D
from ..L2.sdxl_resnet import ResnetBlock2D
from ..L2.sdxl_upsample import Upsample2D
from .sdxl_spatial_transformer import Transformer2DModel


class DownBlock2D(nn.Module):
    """UNet encoder block without cross-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels, padding=downsample_padding),
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    """UNet encoder block with cross-attention transformers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        use_linear_projection: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                )
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, out_channels, padding=downsample_padding),
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    """UNet mid block: ResnetBlock2D + Transformer2DModel + ResnetBlock2D."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
        output_scale_factor: float = 1.0,
        use_linear_projection: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
                output_scale_factor=output_scale_factor,
            ),
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
                output_scale_factor=output_scale_factor,
            ),
        ])

        self.attentions = nn.ModuleList([
            Transformer2DModel(
                num_attention_heads=num_attention_heads,
                attention_head_dim=in_channels // num_attention_heads,
                in_channels=in_channels,
                num_layers=transformer_layers_per_block,
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
            ),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class UpBlock2D(nn.Module):
    """UNet decoder block without cross-attention."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 3,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                )
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, out_channels),
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    """UNet decoder block with cross-attention transformers."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 3,
        transformer_layers_per_block: int = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
        add_upsample: bool = True,
        use_linear_projection: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                )
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, out_channels),
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states
