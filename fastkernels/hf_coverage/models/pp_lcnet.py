"""Default PP-LCNet backbone with explicit hard-activation and product patches."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU

from ..patches.hard_activations import HardSigmoid, HardSwish
from ..patches.product_gate import ProductGate
from ..runner import Workload


class ConvLayer(nn.Module):
    def __init__(self, source, target, kernel=3, stride=1, groups=1, activation=True):
        super().__init__()
        self.convolution = Conv2d(source, target, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False)
        self.normalization = BatchNorm2d(target)
        self.activation = HardSwish() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, width, reduction):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d(keepdim=True)
        self.convolutions = nn.Sequential(Conv2d(width, width // reduction, 1), ReLU(),
                                          Conv2d(width // reduction, width, 1), HardSigmoid())
        self.product = ProductGate()

    def forward(self, x):
        gate = self.convolutions(self.avg_pool(x)).expand_as(x)
        return self.product(torch.cat((x, gate), dim=-1))


def divisible(value, divisor):
    width = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return width + divisor if width < 0.9 * value else width


class Depthwise(nn.Module):
    def __init__(self, config, kernel, source, target, stride, se, conv=ConvLayer):
        super().__init__()
        source, target = [divisible(width * config.scale, config.divisor) for width in (source, target)]
        self.depthwise_convolution = conv(source, source, kernel, stride=stride, groups=source)
        self.squeeze_excitation_module = SqueezeExcitation(source, config.reduction) if se else nn.Identity()
        self.pointwise_convolution = conv(source, target, 1)

    def forward(self, x):
        return self.pointwise_convolution(self.squeeze_excitation_module(self.depthwise_convolution(x)))


class Backbone(nn.Module):
    def __init__(self, config, conv=ConvLayer):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.convolution = ConvLayer(3, divisible(config.stem_channels * config.scale, config.divisor), stride=config.stem_stride)
        self.encoder.blocks = nn.ModuleList()
        for stage in config.block_configs:
            block = nn.Module()
            block.layers = nn.ModuleList([Depthwise(config, *spec, conv=conv) for spec in stage])
            self.encoder.blocks.append(block)
        self.out_indices = tuple(config.out_indices)

    def forward(self, pixel_values):
        hidden = self.encoder.convolution(pixel_values)
        selected = [hidden] if 0 in self.out_indices else []
        for i, block in enumerate(self.encoder.blocks, 1):
            for layer in block.layers:
                hidden = layer(hidden)
            if i in self.out_indices:
                selected.append(hidden)
        return {f"feature_maps.{index}": value for index, value in enumerate(selected)}


def build_from_config(config, device, dtype):
    if config.hidden_act != "hardswish":
        raise ValueError("Preserve default hard-swish activations")
    return Backbone(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
