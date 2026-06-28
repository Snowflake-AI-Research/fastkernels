"""ConvNeXtV2 image classification model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.global_avg_pool2d import GlobalAvgPool2d
from ..L1.layer_norm import LayerNorm
from ..L1.layer_norm2d import LayerNorm2d
from ..L1.linear import Linear
from ..L3.convnextv2_stage import ConvNeXtV2Stage


@dataclass
class ImageClassifierOutput:
    logits: torch.Tensor


class ConvNeXtV2Embeddings(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_embeddings = Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.layernorm = LayerNorm2d(hidden_size, eps=1e-6)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(pixel_values)
        x = self.layernorm(x)
        return x


class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, hidden_sizes: list[int], depths: list[int]):
        super().__init__()
        stages = []
        for i, (dim, depth) in enumerate(zip(hidden_sizes, depths)):
            in_dim = hidden_sizes[i - 1] if i > 0 else dim
            stages.append(
                ConvNeXtV2Stage(
                    in_dim=in_dim,
                    dim=dim,
                    depth=depth,
                    hidden_dim=4 * dim,
                    downsample=i > 0,
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class ConvNeXtV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ConvNeXtV2Embeddings(
            num_channels=config.num_channels,
            hidden_size=config.hidden_sizes[0],
            patch_size=config.patch_size,
        )
        self.encoder = ConvNeXtV2Encoder(
            hidden_sizes=list(config.hidden_sizes),
            depths=list(config.depths),
        )
        self.pooler = GlobalAvgPool2d()
        self.layernorm = LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        pooled = self.pooler(x)
        pooled = self.layernorm(pooled)
        return x, pooled


class ConvNextV2ForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convnextv2 = ConvNeXtV2Model(config)
        self.classifier = Linear(config.hidden_sizes[-1], config.num_labels)

    def forward(self, pixel_values: torch.Tensor) -> ImageClassifierOutput:
        _, pooled = self.convnextv2(pixel_values)
        logits = self.classifier(pooled)
        return ImageClassifierOutput(logits=logits)
