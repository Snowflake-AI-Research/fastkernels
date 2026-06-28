"""RTDetrV2 ResNet backbone layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.avg_pool2d import AvgPool2d
from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.frozen_batch_norm2d import FrozenBatchNorm2d
from ..L1.gelu import GELU
from ..L1.max_pool2d import MaxPool2d
from ..L1.relu import ReLU
from ..L1.silu import SiLU

_ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}


def _activation(name: str) -> nn.Module:
    return _ACTIVATIONS[name.lower()]()


def _norm(num_features: int, frozen: bool) -> nn.Module:
    if frozen:
        return FrozenBatchNorm2d(num_features)
    return BatchNorm2d(num_features)


class RTDetrV2ResNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str | None = "relu",
        frozen_batch_norm: bool = False,
    ):
        super().__init__()
        self.convolution = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = _norm(out_channels, frozen_batch_norm)
        self.activation = _activation(activation) if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        return self.activation(x)


class RTDetrV2ResNetEmbeddings(nn.Module):
    def __init__(self, config, frozen_batch_norm: bool = False):
        super().__init__()
        self.num_channels = config.num_channels
        self.embedder = nn.Sequential(
            RTDetrV2ResNetConvLayer(
                config.num_channels,
                config.embedding_size // 2,
                kernel_size=3,
                stride=2,
                activation=config.hidden_act,
                frozen_batch_norm=frozen_batch_norm,
            ),
            RTDetrV2ResNetConvLayer(
                config.embedding_size // 2,
                config.embedding_size // 2,
                kernel_size=3,
                stride=1,
                activation=config.hidden_act,
                frozen_batch_norm=frozen_batch_norm,
            ),
            RTDetrV2ResNetConvLayer(
                config.embedding_size // 2,
                config.embedding_size,
                kernel_size=3,
                stride=1,
                activation=config.hidden_act,
                frozen_batch_norm=frozen_batch_norm,
            ),
        )
        self.pooler = MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError("Input channels do not match backbone configuration.")
        return self.pooler(self.embedder(pixel_values))


class RTDetrV2ResNetShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, frozen_batch_norm: bool = False):
        super().__init__()
        self.convolution = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = _norm(out_channels, frozen_batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.convolution(x))


class RTDetrV2ResNetBasicLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        should_apply_shortcut: bool = False,
        frozen_batch_norm: bool = False,
    ):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut = (
                nn.Sequential(
                    AvgPool2d(2, 2, 0, ceil_mode=True),
                    RTDetrV2ResNetShortcut(in_channels, out_channels, stride=1, frozen_batch_norm=frozen_batch_norm),
                )
                if should_apply_shortcut
                else nn.Identity()
            )
        else:
            self.shortcut = (
                RTDetrV2ResNetShortcut(in_channels, out_channels, stride=stride, frozen_batch_norm=frozen_batch_norm)
                if should_apply_shortcut
                else nn.Identity()
            )

        self.layer = nn.Sequential(
            RTDetrV2ResNetConvLayer(
                in_channels, out_channels, stride=stride, frozen_batch_norm=frozen_batch_norm
            ),
            RTDetrV2ResNetConvLayer(
                out_channels, out_channels, activation=None, frozen_batch_norm=frozen_batch_norm
            ),
        )
        self.activation = _activation(config.hidden_act)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        hidden_state = hidden_state + self.shortcut(residual)
        return self.activation(hidden_state)


class RTDetrV2ResNetBottleNeckLayer(nn.Module):
    def __init__(self, config, in_channels: int, out_channels: int, stride: int = 1, frozen_batch_norm: bool = False):
        super().__init__()
        reduction = 4
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduced_channels = out_channels // reduction
        if stride == 2:
            self.shortcut = nn.Sequential(
                AvgPool2d(2, 2, 0, ceil_mode=True),
                RTDetrV2ResNetShortcut(
                    in_channels, out_channels, stride=1, frozen_batch_norm=frozen_batch_norm
                )
                if should_apply_shortcut
                else nn.Identity(),
            )
        else:
            self.shortcut = (
                RTDetrV2ResNetShortcut(
                    in_channels, out_channels, stride=stride, frozen_batch_norm=frozen_batch_norm
                )
                if should_apply_shortcut
                else nn.Identity()
            )
        self.layer = nn.Sequential(
            RTDetrV2ResNetConvLayer(
                in_channels,
                reduced_channels,
                kernel_size=1,
                stride=stride if config.downsample_in_bottleneck else 1,
                frozen_batch_norm=frozen_batch_norm,
            ),
            RTDetrV2ResNetConvLayer(
                reduced_channels,
                reduced_channels,
                stride=1 if config.downsample_in_bottleneck else stride,
                frozen_batch_norm=frozen_batch_norm,
            ),
            RTDetrV2ResNetConvLayer(
                reduced_channels,
                out_channels,
                kernel_size=1,
                activation=None,
                frozen_batch_norm=frozen_batch_norm,
            ),
        )
        self.activation = _activation(config.hidden_act)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        hidden_state = hidden_state + self.shortcut(residual)
        return self.activation(hidden_state)


class RTDetrV2ResNetStage(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
        frozen_batch_norm: bool = False,
    ):
        super().__init__()
        layer_cls = RTDetrV2ResNetBottleNeckLayer if config.layer_type == "bottleneck" else RTDetrV2ResNetBasicLayer
        if config.layer_type == "bottleneck":
            first_layer = layer_cls(
                config,
                in_channels,
                out_channels,
                stride=stride,
                frozen_batch_norm=frozen_batch_norm,
            )
        else:
            first_layer = layer_cls(
                config,
                in_channels,
                out_channels,
                stride=stride,
                should_apply_shortcut=True,
                frozen_batch_norm=frozen_batch_norm,
            )
        self.layers = nn.Sequential(
            first_layer,
            *[
                layer_cls(
                    config,
                    out_channels,
                    out_channels,
                    frozen_batch_norm=frozen_batch_norm,
                )
                for _ in range(depth - 1)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
