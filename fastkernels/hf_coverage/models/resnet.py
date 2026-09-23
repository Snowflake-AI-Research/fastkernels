"""Default bottleneck ResNetModel inference, including spatial average pooling."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.rtdetrv2_resnet import (
    RTDetrV2ResNetConvLayer as ResNetConvLayer,
    RTDetrV2ResNetShortcut as ResNetShortcut,
)


class _Bottleneck(nn.Module):
    """HF residual wiring around existing Conv-BN-activation operations.

    The wider RT-DETRv2 bottleneck pools before its shortcut projection.
    HF ResNet instead puts the stride in that projection convolution.
    """

    def __init__(self, config, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        reduced = out_channels // 4
        self.shortcut = (
            ResNetShortcut(in_channels, out_channels, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.layer = nn.Sequential(
            ResNetConvLayer(
                in_channels,
                reduced,
                kernel_size=1,
                stride=stride if config.downsample_in_bottleneck else 1,
            ),
            ResNetConvLayer(
                reduced, reduced, stride=1 if config.downsample_in_bottleneck else stride
            ),
            ResNetConvLayer(reduced, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer(hidden_states)
        return self.activation(hidden_states + self.shortcut(residual))


class _Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = int(config.num_channels)
        self.embedder = ResNetConvLayer(
            self.num_channels, config.embedding_size, kernel_size=7, stride=2
        )
        self.pooler = MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != self.num_channels:
            raise ValueError("Expected NCHW pixel_values with the configured channels")
        return self.pooler(self.embedder(pixel_values))


class _Stage(nn.Module):
    def __init__(self, config, in_channels: int, out_channels: int, stride: int, depth: int):
        super().__init__()
        self.layers = nn.Sequential(
            _Bottleneck(config, in_channels, out_channels, stride=stride),
            *[_Bottleneck(config, out_channels, out_channels) for _ in range(depth - 1)],
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states)


class _Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        in_channels = int(config.embedding_size)
        for index, (out_channels, depth) in enumerate(zip(config.hidden_sizes, config.depths)):
            stride = 2 if index or config.downsample_in_first_stage else 1
            self.stages.append(_Stage(config, in_channels, out_channels, stride, depth))
            in_channels = out_channels

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            hidden_states = stage(hidden_states)
        return hidden_states


class ResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = _Embeddings(config)
        self.encoder = _Encoder(config)
        self.pooler = GlobalAvgPool2d(keepdim=True)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The ResNet coverage model supports inference only")
        hidden_states = self.encoder(self.embedder(pixel_values))
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": self.pooler(hidden_states),
        }


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> ResNetModel:
    if config.hidden_act != "relu" or config.layer_type != "bottleneck":
        raise ValueError("This pilot requires the default ReLU bottleneck graph")
    if len(config.hidden_sizes) != 4 or len(config.depths) != 4:
        raise ValueError("This pilot preserves all four ResNet stages")
    if config.num_channels <= 0 or config.embedding_size <= 0:
        raise ValueError("Input and embedding channel counts must be positive")
    if any(size <= 0 or size % 4 for size in config.hidden_sizes):
        raise ValueError("Bottleneck stage widths must be positive multiples of four")
    if any(depth < 2 for depth in config.depths):
        raise ValueError("Each development stage must retain a repeated residual block")
    if getattr(config, "output_hidden_states", False):
        raise ValueError("This pilot returns the default final feature map and pooler output")
    return ResNetModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model: ResNetModel, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model: ResNetModel, inputs, config) -> dict[str, Workload]:
    del config
    if set(inputs) != {"pixel_values"}:
        raise ValueError("Default ResNet inference expects only pixel_values")
    return {"forward": Workload(run=lambda: model(inputs["pixel_values"]))}
