"""UVDoc's parallel dilated bridges and point-coordinate prediction head."""

import torch
from torch import nn
from torch.nn import functional as F

from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.relu import ReLU


class _PReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.relu, self.product = ReLU(), ProductGate()

    def forward(self, x):
        negative = self.relu(-x)
        return self.relu(x) - self.product(torch.cat((negative, self.weight.expand_as(x)), dim=-1))


class _Conv(Conv2d):
    def __init__(self, incoming, outgoing, kernel, stride=1, padding=0, dilation=1, bias=False, reflect=False):
        super().__init__(incoming, outgoing, kernel, stride, 0 if reflect else padding,
                         dilation=dilation, bias=bias)
        self.reflect_padding = padding if reflect else 0

    def forward(self, x):
        if self.reflect_padding:
            x = F.pad(x, (self.reflect_padding,) * 4, mode="reflect")
        return super().forward(x)


class _ConvLayer(nn.Module):
    def __init__(self, incoming, outgoing, kernel=3, stride=1, padding=0, dilation=1,
                 bias=False, activation="relu", reflect=False):
        super().__init__()
        self.convolution = _Conv(incoming, outgoing, kernel, stride, padding, dilation, bias, reflect)
        self.normalization = BatchNorm2d(outgoing)
        self.activation = _PReLU() if activation == "prelu" else ReLU() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.normalization(self.convolution(x)))


class _Residual(nn.Module):
    def __init__(self, incoming, outgoing, dilation, downsample, kernel):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv_down = (_ConvLayer(incoming, outgoing, kernel, stride, kernel//2, bias=True, activation=None)
                          if downsample else nn.Identity())
        self.conv_start = _ConvLayer(incoming, outgoing, kernel, stride, dilation*2, dilation, bias=True)
        self.conv_final = _ConvLayer(outgoing, outgoing, kernel, 1, dilation*2, dilation, bias=True, activation=None)
        self.activation = ReLU()

    def forward(self, x):
        return self.activation(self.conv_final(self.conv_start(x)) + self.conv_down(x))


class _Layers(nn.Module):
    def __init__(self, layers, name):
        super().__init__()
        self.name = name
        self.add_module(name, nn.ModuleList(layers))

    def forward(self, x):
        for layer in getattr(self, self.name):
            x = layer(x)
        return x


class UVDocModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        b = c.backbone_config
        self.out_indices = b.out_indices
        self.backbone = nn.Module()
        self.backbone.resnet = nn.Module()
        self.backbone.resnet.resnet_head = nn.ModuleList([
            _ConvLayer(incoming, outgoing, b.kernel_size, 2, b.kernel_size//2) for incoming, outgoing in b.resnet_head])
        self.backbone.resnet.resnet_down = nn.ModuleList([
            _Layers([_Residual(*spec, b.kernel_size) for spec in stage], "layers") for stage in b.resnet_configs])
        self.backbone.bridge = nn.Module()
        self.backbone.bridge.bridge = nn.ModuleList([
            _Layers([_ConvLayer(width, width, padding=dilation, dilation=dilation) for width, dilation in stage], "blocks")
            for stage in b.stage_configs])
        self.head = nn.Module()
        self.head.bridge_connector = _ConvLayer(c.bridge_connector[0]*len(b.stage_configs), c.bridge_connector[1], 1)
        self.head.out_point_positions2D = nn.Module()
        incoming, outgoing = c.out_point_positions2D[0]
        self.head.out_point_positions2D.conv_down = _ConvLayer(incoming, outgoing, c.kernel_size,
            padding=c.kernel_size//2, activation=c.hidden_act, reflect=True)
        incoming, outgoing = c.out_point_positions2D[1]
        self.head.out_point_positions2D.conv_up = _Conv(incoming, outgoing, c.kernel_size,
            padding=c.kernel_size//2, bias=True, reflect=True)

    def forward(self, pixel_values):
        hidden = pixel_values
        for layer in self.backbone.resnet.resnet_head:
            hidden = layer(hidden)
        for stage in self.backbone.resnet.resnet_down:
            hidden = stage(hidden)
        # Native bridge stages consume the same backbone output independently.
        states = [hidden] + [stage(hidden) for stage in self.backbone.bridge.bridge]
        joined = torch.cat([states[i] for i in self.out_indices], dim=1)
        hidden = self.head.bridge_connector(joined)
        hidden = self.head.out_point_positions2D.conv_up(self.head.out_point_positions2D.conv_down(hidden))
        return {"last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.padding_mode != "reflect" or config.hidden_act != "prelu":
        raise ValueError("The declared UVDoc checkpoint uses reflect padding and PReLU")
    return UVDocModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
