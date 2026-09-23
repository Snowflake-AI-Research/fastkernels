"""MobileNetV1's depthwise-separable encoder and default spatial pooler."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.hf_coverage.patches.mobilenet_relu6 import ReLU6
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L2.mobilenetv4_uib import ConvNormAct


class _MobileConv(ConvNormAct):
    """Existing Conv-BN-activation with HF's SAME padding and ReLU6 choice."""

    def __init__(self, config, in_channels, out_channels, kernel_size=1, stride=1,
                 groups=1, activation=True, momentum=0.9997):
        super().__init__(in_channels, out_channels, kernel_size, stride, groups, apply_act=False)
        self.kernel_size = kernel_size
        self.tf_padding = config.tf_padding
        self.pad = Pad()
        if self.tf_padding:
            self.conv.padding = (0, 0)
        self.bn = BatchNorm2d(out_channels, eps=config.layer_norm_eps, momentum=momentum)
        self.act = ReLU6() if activation else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tf_padding:
            height, width = hidden_states.shape[-2:]
            stride_height, stride_width = self.conv.stride
            pad_height = max(self.kernel_size - (height % stride_height or stride_height), 0)
            pad_width = max(self.kernel_size - (width % stride_width or stride_width), 0)
            hidden_states = self.pad(hidden_states, (
                pad_width // 2, pad_width - pad_width // 2,
                pad_height // 2, pad_height - pad_height // 2,
            ))
        return super().forward(hidden_states)


class MobileNetV1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        depth = 32
        out_channels = max(int(depth * config.depth_multiplier), config.min_depth)
        self.conv_stem = _MobileConv(config, config.num_channels, out_channels, 3, stride=2)
        self.layer = nn.ModuleList()
        for index, stride in enumerate((1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1)):
            in_channels = out_channels
            if stride == 2 or index == 0:
                depth *= 2
                out_channels = max(int(depth * config.depth_multiplier), config.min_depth)
            self.layer.append(_MobileConv(config, in_channels, in_channels, 3, stride, groups=in_channels))
            self.layer.append(_MobileConv(config, in_channels, out_channels))
        self.pooler = GlobalAvgPool2d()

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The MobileNet coverage models support inference only")
        hidden_states = self.conv_stem(pixel_values)
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler(hidden_states)}


def _check_config(config) -> None:
    if config.hidden_act != "relu6":
        raise ValueError("These MobileNet pilots preserve the default ReLU6 activation")
    if config.num_channels <= 0 or config.depth_multiplier <= 0 or config.min_depth <= 0:
        raise ValueError("Input channels and width settings must be positive")
    if getattr(config, "output_hidden_states", False):
        raise ValueError("These pilots return the default feature map and pooler output")


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> MobileNetV1Model:
    _check_config(config)
    return MobileNetV1Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {
        name.replace(".convolution.", ".conv.").replace(".normalization.", ".bn."): value
        for name, value in state_dict.items()
    }
    model.load_state_dict(mapped, strict=True)
