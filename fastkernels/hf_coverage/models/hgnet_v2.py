"""HGNetV2's default four-stage backbone from convolution, normalization and pool ops."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from ..runner import Workload


class Conv(nn.Module):
    def __init__(self, source, target, kernel, stride=1, groups=1, activation=True):
        super().__init__()
        self.convolution = Conv2d(source, target, kernel, stride=stride, groups=groups,
                                  padding=(kernel - 1) // 2, bias=False)
        self.normalization = BatchNorm2d(target)
        self.activation = ReLU() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class LightConv(nn.Module):
    def __init__(self, source, target, kernel):
        super().__init__()
        self.conv1 = Conv(source, target, 1, activation=False)
        self.conv2 = Conv(target, target, kernel, groups=target)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Stem(nn.Module):
    def __init__(self, config):
        super().__init__()
        a, b, c = config.stem_channels
        strides = config.stem_strides
        self.stem1 = Conv(a, b, 3, strides[0])
        self.stem2a = Conv(b, b // 2, 2, strides[1])
        self.stem2b = Conv(b // 2, b, 2, strides[2])
        self.stem3 = Conv(2 * b, b, 3, strides[3])
        self.stem4 = Conv(b, c, 1, strides[4])
        self.pad = Pad()
        self.pool = MaxPool2d(2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.pad(self.stem1(x), (0, 1, 0, 1))
        branch = self.stem2b(self.pad(self.stem2a(x), (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat((self.pool(x), branch), dim=1)))


class Block(nn.Module):
    def __init__(self, source, middle, target, count, kernel, light, residual):
        super().__init__()
        conv = LightConv if light else Conv
        self.layers = nn.ModuleList([conv(source if i == 0 else middle, middle, kernel)
                                     for i in range(count)])
        self.aggregation = nn.Sequential(Conv(source + count * middle, target // 2, 1),
                                         Conv(target // 2, target, 1))
        self.residual = residual

    def forward(self, x):
        values = [x]
        for layer in self.layers:
            values.append(layer(values[-1]))
        hidden = self.aggregation(torch.cat(values, dim=1))
        return x + hidden if self.residual else hidden


class Stage(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        source, middle, target = [getattr(config, field)[index] for field in
                                  ("stage_in_channels", "stage_mid_channels", "stage_out_channels")]
        self.downsample = (Conv(source, source, 3, config.stage_downsample_strides[index],
                                groups=source, activation=False)
                           if config.stage_downsample[index] else nn.Identity())
        self.blocks = nn.ModuleList([Block(source if i == 0 else target, middle, target,
                                           config.stage_numb_of_layers[index], config.stage_kernel_size[index],
                                           config.stage_light_block[index], i != 0)
                                      for i in range(config.stage_num_blocks[index])])

    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = Stem(config)
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList([Stage(config, i) for i in range(len(config.stage_in_channels))])
        self.out_indices = tuple(config.out_indices)

    def forward(self, pixel_values):
        x = self.embedder(pixel_values)
        values = [x] if 0 in self.out_indices else []
        for i, stage in enumerate(self.encoder.stages, 1):
            x = stage(x)
            if i in self.out_indices:
                values.append(x)
        return {f"feature_maps.{i}": value for i, value in enumerate(values)}


def build_from_config(config, device, dtype):
    if config.hidden_act != "relu" or config.use_learnable_affine_block:
        raise ValueError("This case preserves the default ReLU without learned affine blocks")
    return Backbone(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
