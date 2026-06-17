"""RTDetrV2 model (L3 composite)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.embedding import Embedding
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L2.rtdetrv2_mlp_head import RTDetrV2MLPPredictionHead
from .rtdetrv2_backbone import RTDetrV2ConvEncoder
from .rtdetrv2_decoder import RTDetrV2Decoder
from .rtdetrv2_hybrid_encoder import RTDetrV2HybridEncoder


class RTDetrV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = RTDetrV2ConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        self.encoder_input_proj = nn.ModuleList()
        for in_channels in intermediate_channel_sizes:
            self.encoder_input_proj.append(
                nn.Sequential(
                    Conv2d(in_channels, config.encoder_hidden_dim, kernel_size=1, bias=False),
                    BatchNorm2d(config.encoder_hidden_dim),
                )
            )

        self.encoder = RTDetrV2HybridEncoder(config)

        if config.num_denoising > 0:
            self.denoising_class_embed = Embedding(config.num_labels + 1, config.d_model, padding_idx=config.num_labels)

        if config.learn_initial_query:
            self.weight_embedding = Embedding(config.num_queries, config.d_model)

        self.enc_output = nn.Sequential(
            Linear(config.d_model, config.d_model),
            LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.enc_score_head = Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = RTDetrV2MLPPredictionHead(config, config.d_model, config.d_model, 4, num_layers=3)

        self.decoder_input_proj = nn.ModuleList()
        in_channels = config.decoder_in_channels[-1]
        for decoder_in in config.decoder_in_channels:
            self.decoder_input_proj.append(
                nn.Sequential(
                    Conv2d(decoder_in, config.d_model, kernel_size=1, bias=False),
                    BatchNorm2d(config.d_model, eps=config.batch_norm_eps),
                )
            )
        for _ in range(config.num_feature_levels - len(config.decoder_in_channels)):
            self.decoder_input_proj.append(
                nn.Sequential(
                    Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                    BatchNorm2d(config.d_model, eps=config.batch_norm_eps),
                )
            )
            in_channels = config.d_model

        self.decoder = RTDetrV2Decoder(config)

    def generate_anchors(self, spatial_shapes, grid_size=0.05, device="cpu", dtype=torch.float32):
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device).to(dtype),
                torch.arange(end=width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.finfo(dtype).max, dtype=dtype, device=device))
        return anchors, valid_mask

    def forward(self, pixel_values, pixel_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=device)

        features = self.backbone(pixel_values, pixel_mask)
        proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]
        encoder_outputs = self.encoder(
            proj_feats,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sources = []
        for level, source in enumerate(encoder_outputs.last_hidden_state):
            sources.append(self.decoder_input_proj[level](source))
        if self.config.num_feature_levels > len(sources):
            source = encoder_outputs.last_hidden_state[-1]
            for i in range(len(sources), self.config.num_feature_levels):
                source = self.decoder_input_proj[i](source)
                sources.append(source)

        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            h, w = source.shape[-2:]
            spatial_shapes[level, 0] = h
            spatial_shapes[level, 1] = w
            spatial_shapes_list.append((h, w))
            source_flatten.append(source.flatten(2).transpose(1, 2))
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        anchors, valid_mask = self.generate_anchors(spatial_shapes_list, device=device, dtype=source_flatten.dtype)
        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)
        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1]),
        )
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])).detach()
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=None,
            reference_points=reference_points_unact.detach(),
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if return_dict:
            return SimpleNamespace(
                last_hidden_state=decoder_outputs.last_hidden_state,
                intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
                intermediate_logits=decoder_outputs.intermediate_logits,
                intermediate_reference_points=decoder_outputs.intermediate_reference_points,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            )
        return (
            decoder_outputs.last_hidden_state,
            decoder_outputs.intermediate_hidden_states,
            decoder_outputs.intermediate_logits,
            decoder_outputs.intermediate_reference_points,
        )
