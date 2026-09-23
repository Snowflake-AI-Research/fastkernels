"""MobileViT's convolution/transformer stages, expanded features and pooler."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L2.efficientnetv2_inverted_residual import InvertedResidual
from fastkernels.tasks.baseline.L2.mobilenetv4_uib import ConvNormAct, _make_divisible
from fastkernels.tasks.baseline.L2.sdxl_attention import SDXLAttention
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from .pvt_v2 import _EagerAttentionCore
from .vit_msn import make_workloads


def _conv(in_channels, out_channels, kernel=1, stride=1, groups=1,
          normalization=True, activation=True, bias=False):
    layer = ConvNormAct(in_channels, out_channels, kernel, stride, groups, apply_act=False)
    if bias:
        layer.conv = Conv2d(in_channels, out_channels, kernel, stride=stride,
                            padding=kernel // 2, groups=groups, bias=True)
    if not normalization:
        layer.bn = nn.Identity()
    layer.act = SiLU() if activation else nn.Identity()
    return layer


def _inverted(config, in_channels, out_channels, stride):
    expanded = _make_divisible(int(round(in_channels * config.expand_ratio)))
    block = InvertedResidual(in_channels, expanded, out_channels, stride,
                             se_reduce_chs=1, has_skip=stride == 1 and in_channels == out_channels)
    block.se = nn.Identity()
    return block


def _mobile_stage(config, in_channels, out_channels, stride, depth):
    return nn.Sequential(*[
        _inverted(config, in_channels if index == 0 else out_channels, out_channels,
                  stride if index == 0 else 1)
        for index in range(depth)
    ])


def _unfold(features, patch):
    batch, channels, height, width = features.shape
    if height % patch or width % patch:
        raise ValueError("Nonoverlapping patch layout requires divisible spatial dimensions")
    return features.reshape(batch, channels, height // patch, patch, width // patch, patch).permute(
        0, 1, 3, 5, 2, 4).reshape(batch, channels, patch * patch, -1)


def _fold(patches, height, width, patch):
    batch, channels, _, _ = patches.shape
    return patches.reshape(batch, channels, patch, patch, height // patch, width // patch).permute(
        0, 1, 4, 2, 5, 3).reshape(batch, channels, height, width)


class _Attention(SDXLAttention):
    def __init__(self, width, heads):
        super().__init__(width, heads=heads, dim_head=width // heads)
        for name in ("to_q", "to_k", "to_v"):
            setattr(self, name, Linear(width, width))
        self.attn = _EagerAttentionCore()

    def forward(self, hidden_states, attn_mask=None):
        if attn_mask is not None:
            raise ValueError("The default MobileViT patch attention is unmasked")
        return super().forward(hidden_states)


class _MobileViTLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, width, depth):
        super().__init__()
        self.patch = config.patch_size
        self.downsampling_layer = _inverted(config, in_channels, out_channels, 2)
        self.conv_kxk = _conv(out_channels, out_channels, config.conv_kernel_size)
        self.conv_1x1 = _conv(out_channels, width, normalization=False, activation=False)
        blocks = []
        for _ in range(depth):
            block = VitEncoderBlock(width, config.num_attention_heads, config.mlp_ratio,
                                    norm_eps=config.layer_norm_eps, proj_drop=config.hidden_dropout_prob)
            block.attn = _Attention(width, config.num_attention_heads)
            block.mlp.act = SiLU()
            blocks.append(block)
        self.transformer = nn.Sequential(*blocks)
        self.layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.conv_projection = _conv(width, out_channels)
        self.fusion = _conv(2 * out_channels, out_channels, config.conv_kernel_size)
        self.resize = Interpolate()

    def forward(self, features):
        residual = self.downsampling_layer(features)
        features = self.conv_1x1(self.conv_kxk(residual))
        batch, channels, height, width = features.shape
        new_height = (height + self.patch - 1) // self.patch * self.patch
        new_width = (width + self.patch - 1) // self.patch * self.patch
        if (height, width) != (new_height, new_width):
            features = self.resize(features, size=(new_height, new_width), mode="bilinear", align_corners=False)
        patches = _unfold(features, self.patch).permute(0, 2, 3, 1).reshape(batch * self.patch ** 2, -1, channels)
        patches = self.layernorm(self.transformer(patches))
        patches = patches.reshape(batch, self.patch ** 2, -1, channels).permute(0, 3, 1, 2)
        features = _fold(patches, new_height, new_width, self.patch)
        if (height, width) != (new_height, new_width):
            features = self.resize(features, size=(height, width), mode="bilinear", align_corners=False)
        return self.fusion(torch.cat((residual, self.conv_projection(features)), dim=1))


class MobileViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.neck_hidden_sizes
        self.conv_stem = _conv(config.num_channels, channels[0], 3, 2)
        self.encoder = nn.ModuleList([
            _mobile_stage(config, channels[0], channels[1], 1, 1),
            _mobile_stage(config, channels[1], channels[2], 2, 3),
            *[_MobileViTLayer(config, channels[index + 2], channels[index + 3], width, depth)
              for index, (width, depth) in enumerate(zip(config.hidden_sizes, (2, 4, 3)))],
        ])
        self.conv_1x1_exp = _conv(channels[5], channels[6])
        self.pooler = GlobalAvgPool2d()

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This MobileViT coverage model supports inference only")
        hidden_states = self.conv_stem(pixel_values)
        for stage in self.encoder:
            hidden_states = stage(hidden_states)
        hidden_states = self.conv_1x1_exp(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler(hidden_states)}


def build_from_config(config, device, dtype):
    if config.output_stride != 32 or config.hidden_act != "silu" or not config.qkv_bias:
        raise ValueError("This case retains the default stride-32 SiLU encoder with biased QKV")
    return MobileViTModel(config).to(device=device, dtype=dtype).eval()


def _state_name(name):
    name = name.replace(".layer.", ".")
    for origin, conv, norm in (("expand_1x1", "conv_pw", "bn1"), ("conv_3x3", "conv_dw", "bn2"),
                               ("reduce_1x1", "conv_pwl", "bn3")):
        name = name.replace(f".{origin}.convolution.", f".{conv}.")
        name = name.replace(f".{origin}.normalization.", f".{norm}.")
    name = name.replace(".convolution.", ".conv.").replace(".normalization.", ".bn.")
    name = name.replace(".layernorm_before.", ".norm1.").replace(".layernorm_after.", ".norm2.")
    for origin, target in (("query", "to_q"), ("key", "to_k"), ("value", "to_v")):
        name = name.replace(f".attention.attention.{origin}.", f".attn.{target}.")
    name = name.replace(".attention.output.dense.", ".attn.to_out.0.")
    return name.replace(".intermediate.dense.", ".mlp.fc1.").replace(".output.dense.", ".mlp.fc2.")


def load_state_dict_into(model, state_dict, config):
    del config
    model.load_state_dict({_state_name(name): value for name, value in state_dict.items()}, strict=True)
