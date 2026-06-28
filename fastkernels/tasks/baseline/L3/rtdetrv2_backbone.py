"""Native RTDetrV2 convolutional backbone."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from ..L1.interpolate import Interpolate
from ..L2.rtdetrv2_resnet import RTDetrV2ResNetEmbeddings, RTDetrV2ResNetStage


class RTDetrV2ResNetEncoder(nn.Module):
    def __init__(self, backbone_config, frozen_batch_norm: bool = False):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                RTDetrV2ResNetStage(
                    backbone_config,
                    backbone_config.embedding_size,
                    backbone_config.hidden_sizes[0],
                    stride=2 if backbone_config.downsample_in_first_stage else 1,
                    depth=backbone_config.depths[0],
                    frozen_batch_norm=frozen_batch_norm,
                )
            ]
        )
        for (in_channels, out_channels), depth in zip(
            zip(backbone_config.hidden_sizes, backbone_config.hidden_sizes[1:]), backbone_config.depths[1:]
        ):
            self.stages.append(
                RTDetrV2ResNetStage(
                    backbone_config,
                    in_channels,
                    out_channels,
                    depth=depth,
                    frozen_batch_norm=frozen_batch_norm,
                )
            )

    def forward(self, hidden_state: torch.Tensor):
        hidden_states = ()
        for stage_module in self.stages:
            hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage_module(hidden_state)
        hidden_states = hidden_states + (hidden_state,)
        return SimpleNamespace(last_hidden_state=hidden_state, hidden_states=hidden_states)


class RTDetrV2ResNetBackbone(nn.Module):
    def __init__(self, backbone_config, frozen_batch_norm: bool = False):
        super().__init__()
        self.stage_names = tuple(backbone_config.stage_names)
        self.out_features = tuple(backbone_config.out_features)
        self.channels = [
            backbone_config.hidden_sizes[idx - 1]
            for idx in backbone_config.out_indices
        ]
        self.embedder = RTDetrV2ResNetEmbeddings(backbone_config, frozen_batch_norm=frozen_batch_norm)
        self.encoder = RTDetrV2ResNetEncoder(backbone_config, frozen_batch_norm=frozen_batch_norm)

    def forward(self, pixel_values: torch.Tensor):
        embedding_output = self.embedder(pixel_values)
        outputs = self.encoder(embedding_output)
        hidden_states = outputs.hidden_states
        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps = feature_maps + (hidden_states[idx],)
        return SimpleNamespace(feature_maps=feature_maps, hidden_states=hidden_states)


class RTDetrV2ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = RTDetrV2ResNetBackbone(
            config.backbone_config,
            frozen_batch_norm=bool(config.freeze_backbone_batch_norms),
        )
        self.intermediate_channel_sizes = self.model.channels
        self._interpolate = Interpolate()

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        features = self.model(pixel_values).feature_maps
        out = []
        for feature_map in features:
            mask = self._interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out
