"""EfficientNet's MBConv encoder and default ceil-mode spatial pooler."""

from __future__ import annotations

import math

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L2.batch_norm_act2d import BatchNormAct2d
from fastkernels.tasks.baseline.L2.efficientnetv2_inverted_residual import InvertedResidual


def _round_filters(config, channels):
    scaled = channels * config.width_coefficient
    divisor = config.depth_divisor
    rounded = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * scaled else rounded


class _PaddedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, groups=1, adjust=True):
        super().__init__()
        self.pad = Pad()
        half = kernel // 2
        self.padding = (half - int(adjust), half, half - int(adjust), half) if stride == 2 else None
        self.conv = Conv2d(in_channels, out_channels, kernel, stride=stride,
                           padding=0 if stride == 2 else half, groups=groups, bias=False)

    def forward(self, hidden_states):
        if self.padding is not None:
            hidden_states = self.pad(hidden_states, self.padding)
        return self.conv(hidden_states)


def _batch_norm(config, width, activation=False, momentum=None):
    return BatchNormAct2d(width, eps=config.batch_norm_eps,
                         momentum=config.batch_norm_momentum if momentum is None else momentum,
                         act_layer=SiLU() if activation else None)


def _block(config, in_channels, out_channels, kernel, stride, expansion, first, index):
    expanded = in_channels * expansion
    block = InvertedResidual(in_channels, expanded, out_channels, stride,
                             max(1, int(in_channels * config.squeeze_expansion_ratio)),
                             has_skip=stride == 1 and not first)
    if expansion == 1:
        block.conv_pw = nn.Identity()
        block.bn1 = nn.Identity()
    else:
        block.bn1 = _batch_norm(config, expanded, activation=True, momentum=0.1)
    block.conv_dw = _PaddedConv(expanded, expanded, kernel, stride, groups=expanded,
                                adjust=index not in config.depthwise_padding)
    block.bn2 = _batch_norm(config, expanded, activation=True)
    block.bn3 = _batch_norm(config, out_channels)
    return block


class EfficientNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        stem_width = _round_filters(config, 32)
        self.stem = _PaddedConv(config.num_channels, stem_width, 3, 2)
        self.stem_bn = _batch_norm(config, stem_width, activation=True)
        self.blocks = nn.ModuleList()
        for stage, repeats in enumerate(config.num_block_repeats):
            in_channels = _round_filters(config, config.in_channels[stage])
            out_channels = _round_filters(config, config.out_channels[stage])
            for index in range(math.ceil(repeats * config.depth_coefficient)):
                first = index == 0
                self.blocks.append(_block(config, in_channels if first else out_channels, out_channels,
                                          config.kernel_sizes[stage], config.strides[stage] if first else 1,
                                          config.expand_ratios[stage], first, len(self.blocks)))
        self.top_conv = Conv2d(out_channels, _round_filters(config, 1280), 1, bias=False)
        self.top_bn = _batch_norm(config, config.hidden_dim, activation=True)
        self.pooler = AvgPool2d(config.hidden_dim, ceil_mode=True)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        hidden_states = self.stem_bn(self.stem(pixel_values))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.top_bn(self.top_conv(hidden_states))
        pooled = self.pooler(hidden_states).reshape(hidden_states.shape[:2])
        return {"last_hidden_state": hidden_states, "pooler_output": pooled}


def build_from_config(config, device, dtype):
    if config.hidden_act != "swish" or config.pooling_type != "mean":
        raise ValueError("This pilot preserves default SiLU and mean pooling")
    if len(config.num_block_repeats) != 7 or config.hidden_dim != _round_filters(config, 1280):
        raise ValueError("All seven stages and the rounded top width must be retained")
    if getattr(config, "output_hidden_states", False):
        raise ValueError("This pilot returns the default feature map and pooler")
    return EfficientNetModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    mapped = {}
    for name, value in state_dict.items():
        for origin, target in (("embeddings.convolution.", "stem.conv."),
                               ("embeddings.batchnorm.", "stem_bn."),
                               ("encoder.blocks.", "blocks."),
                               ("encoder.top_conv.", "top_conv."),
                               ("encoder.top_bn.", "top_bn."),
                               (".expansion.expand_conv.", ".conv_pw."),
                               (".expansion.expand_bn.", ".bn1."),
                               (".depthwise_conv.depthwise_conv.", ".conv_dw.conv."),
                               (".depthwise_conv.depthwise_norm.", ".bn2."),
                               (".squeeze_excite.reduce.", ".se.conv_reduce."),
                               (".squeeze_excite.expand.", ".se.conv_expand."),
                               (".projection.project_conv.", ".conv_pwl."),
                               (".projection.project_bn.", ".bn3.")):
            name = name.replace(origin, target)
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
