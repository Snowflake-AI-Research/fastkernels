"""PP-LCNetV3 preserves every default reparameterization branch during inference."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d

from .pp_lcnet import Backbone, ConvLayer, load_state_dict_into, make_workloads
from ..patches.hard_activations import HardSwish


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Both learned parameters are scalar constants during inference.
        return self.scale * x + self.bias


class ActAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = HardSwish()
        self.lab = Affine()

    def forward(self, x):
        return self.lab(self.act(x))


class RepLayer(nn.Module):
    def __init__(self, source, target, kernel, stride=1, groups=1, branches=4):
        super().__init__()
        self.stride = stride
        self.identity = BatchNorm2d(source) if source == target and stride == 1 else None
        self.conv_symmetric = nn.ModuleList([
            ConvLayer(source, target, kernel, stride, groups, activation=False) for _ in range(branches)
        ])
        self.conv_small_symmetric = ConvLayer(source, target, 1, stride, groups, activation=False) if kernel > 1 else None
        self.lab = Affine()
        self.act = ActAffine()

    def forward(self, x):
        result = self.identity(x) if self.identity is not None else None
        if self.conv_small_symmetric is not None:
            value = self.conv_small_symmetric(x)
            result = value if result is None else result + value
        for branch in self.conv_symmetric:
            value = branch(x)
            result = value if result is None else result + value
        result = self.lab(result)
        return result if self.stride == 2 else self.act(result)


def build_from_config(config, device, dtype):
    if config.hidden_act != "hardswish":
        raise ValueError("Preserve default hard-swish and affine activation blocks")
    def conv(source, target, kernel, stride=1, groups=1):
        return RepLayer(source, target, kernel, stride, groups, branches=config.conv_symmetric_num)
    model = Backbone(config, conv=conv)
    # V3's stem is convolution plus normalization; only later blocks activate.
    model.encoder.convolution.activation = nn.Identity()
    return model.to(device=device, dtype=dtype).eval()
