"""MobileNetV2's default inverted-residual encoder and spatial pooler."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.mobilenet_v1 import _MobileConv, _check_config, make_workloads
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L2.mobilenetv4_uib import UniversalInvertedResidual


def _depth(config, channels):
    value = int(round(channels * config.depth_multiplier))
    result = max(config.min_depth, int(value + config.depth_divisible_by / 2)
                 // config.depth_divisible_by * config.depth_divisible_by)
    return result + config.depth_divisible_by if result < 0.9 * value else result


def _conv(config, in_channels, out_channels, kernel_size=1, stride=1, groups=1, activation=True):
    return _MobileConv(config, in_channels, out_channels, kernel_size, stride,
                       groups, activation, momentum=0.997)


def _inverted_residual(config, in_channels, out_channels, stride):
    block = UniversalInvertedResidual(in_channels, out_channels, stride=stride,
                                     exp_ratio=config.expand_ratio)
    # HF's expansion rounding is independent of the model width multiplier.
    value = int(round(in_channels * config.expand_ratio))
    expanded = max(config.min_depth, int(value + config.depth_divisible_by / 2)
                   // config.depth_divisible_by * config.depth_divisible_by)
    if expanded < 0.9 * value:
        expanded += config.depth_divisible_by
    block.pw_exp = _conv(config, in_channels, expanded)
    block.dw_mid = _conv(config, expanded, expanded, 3, stride, groups=expanded)
    block.pw_proj = _conv(config, expanded, out_channels, activation=False)
    return block


class _Stem(nn.Module):
    def __init__(self, config, out_channels):
        super().__init__()
        expanded = _depth(config, 32)
        self.first_conv = _conv(config, config.num_channels, expanded, 3, stride=2)
        self.conv_3x3 = _conv(config, expanded, expanded, 3, groups=expanded)
        self.reduce_1x1 = _conv(config, expanded, out_channels, activation=False)

    def forward(self, hidden_states):
        return self.reduce_1x1(self.conv_3x3(self.first_conv(hidden_states)))


class MobileNetV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = [_depth(config, value) for value in
                    (16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320)]
        strides = (2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1)
        self.conv_stem = _Stem(config, channels[0])
        self.layer = nn.ModuleList([
            _inverted_residual(config, channels[index], channels[index + 1], stride)
            for index, stride in enumerate(strides)
        ])
        output_channels = 1280 if config.finegrained_output and config.depth_multiplier < 1 else _depth(config, 1280)
        self.conv_1x1 = _conv(config, channels[-1], output_channels)
        self.pooler = GlobalAvgPool2d()

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The MobileNet coverage models support inference only")
        hidden_states = self.conv_stem(pixel_values)
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        hidden_states = self.conv_1x1(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler(hidden_states)}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> MobileNetV2Model:
    _check_config(config)
    if config.output_stride != 32 or not config.first_layer_is_expansion:
        raise ValueError("This pilot preserves the default stride-32 encoder and expansion stem")
    if config.depth_divisible_by <= 0 or config.expand_ratio <= 0:
        raise ValueError("Channel divisor and expansion ratio must be positive")
    return MobileNetV2Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace(".convolution.", ".conv.").replace(".normalization.", ".bn.")
        if name.startswith("layer."):
            name = name.replace(".expand_1x1.", ".pw_exp.")
            name = name.replace(".conv_3x3.", ".dw_mid.").replace(".reduce_1x1.", ".pw_proj.")
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
