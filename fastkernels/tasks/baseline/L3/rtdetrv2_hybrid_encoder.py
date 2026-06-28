"""RTDetrV2 hybrid encoder."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from ..L1.interpolate import Interpolate
from ..L2.rtdetrv2_conv_norm import RTDetrV2ConvNormLayer
from ..L2.rtdetrv2_csp_rep_layer import RTDetrV2CSPRepLayer
from ..L2.rtdetrv2_encoder_layer import RTDetrV2EncoderLayer


class RTDetrV2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([RTDetrV2EncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, src, src_mask=None, pos_embed=None, output_attentions: bool = False):
        hidden_states = src
        attns = ()
        for layer in self.layers:
            outputs = layer(
                hidden_states,
                attention_mask=src_mask,
                position_embeddings=pos_embed,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if output_attentions:
                attns += (outputs[1],)
        if output_attentions:
            return hidden_states, attns
        return (hidden_states,)


class RTDetrV2HybridEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1
        activation = config.activation_function

        self._upsample = Interpolate()
        self.encoder = nn.ModuleList([RTDetrV2Encoder(config) for _ in range(len(self.encode_proj_layers))])
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_fpn_stages):
            self.lateral_convs.append(
                RTDetrV2ConvNormLayer(
                    config,
                    in_channels=self.encoder_hidden_dim,
                    out_channels=self.encoder_hidden_dim,
                    kernel_size=1,
                    stride=1,
                    activation=activation,
                )
            )
            self.fpn_blocks.append(RTDetrV2CSPRepLayer(config))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_pan_stages):
            self.downsample_convs.append(
                RTDetrV2ConvNormLayer(
                    config,
                    in_channels=self.encoder_hidden_dim,
                    out_channels=self.encoder_hidden_dim,
                    kernel_size=3,
                    stride=2,
                    activation=activation,
                )
            )
            self.pan_blocks.append(RTDetrV2CSPRepLayer(config))

    @staticmethod
    def build_2d_sincos_position_embedding(
        width, height, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ):
        grid_w = torch.arange(width, device=device).to(dtype)
        grid_h = torch.arange(height, device=device).to(dtype)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, device=device).to(dtype) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        del attention_mask, position_embeddings, spatial_shapes, level_start_index, valid_ratios
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        hidden_states = inputs_embeds

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states[enc_ind],)
                height, width = hidden_states[enc_ind].shape[2:]
                src_flatten = hidden_states[enc_ind].flatten(2).permute(0, 2, 1)
                pos_embed = self.build_2d_sincos_position_embedding(
                    width,
                    height,
                    self.encoder_hidden_dim,
                    self.positional_encoding_temperature,
                    device=src_flatten.device,
                    dtype=src_flatten.dtype,
                )
                layer_outputs = self.encoder[i](src_flatten, pos_embed=pos_embed, output_attentions=output_attentions)
                hidden_states[enc_ind] = layer_outputs[0].permute(0, 2, 1).reshape(
                    -1, self.encoder_hidden_dim, height, width
                ).contiguous()
                if output_attentions:
                    all_attentions = all_attentions + layer_outputs[1]
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states[enc_ind],)

        fpn_feature_maps = [hidden_states[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = hidden_states[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = lateral_conv(fpn_feature_maps[-1])
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = self._upsample(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            fpn_feature_maps.append(fpn_block(fused_feature_map))
        fpn_feature_maps = fpn_feature_maps[::-1]

        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            pan_feature_maps.append(pan_block(fused_feature_map))

        if return_dict:
            return SimpleNamespace(last_hidden_state=pan_feature_maps, hidden_states=encoder_states, attentions=all_attentions)
        return pan_feature_maps, encoder_states, all_attentions
