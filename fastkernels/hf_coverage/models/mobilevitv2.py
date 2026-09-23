"""MobileViTV2's separable linear patch attention and default feature pooler."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L1.linear import BMM
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.mobilenetv4_uib import _make_divisible
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from ..patches.product_gate import ProductGate
from .mobilevit import _conv, _fold, _inverted, _mobile_stage, _state_name as _mobile_state, _unfold
from .vit_msn import make_workloads


class _LinearAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.qkv_proj = _conv(width, 1 + 2 * width, normalization=False, activation=False, bias=True)
        self.out_proj = _conv(width, width, normalization=False, activation=False, bias=True)
        self.softmax = Softmax(dim=-1)
        self.matmul = BMM()
        self.relu = ReLU()
        self.gate = ProductGate()

    def forward(self, hidden_states, attn_mask=None):
        if attn_mask is not None:
            raise ValueError("The default MobileViTV2 linear attention is unmasked")
        query, key, value = self.qkv_proj(hidden_states).split((1, self.width, self.width), dim=1)
        scores = self.softmax(query)
        # Genuine weighted reduction: [B,P,1,N] @ [B,P,N,C], without expansion.
        context = self.matmul(scores.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        values = self.relu(value)
        packed = torch.cat((values, context.expand_as(values)), dim=1).flatten(1)
        output = self.gate(packed).reshape(values.shape)
        return self.out_proj(output)


class _MobileViTV2Layer(nn.Module):
    def __init__(self, config, in_channels, out_channels, width, depth):
        super().__init__()
        self.patch = config.patch_size
        self.downsampling_layer = _inverted(config, in_channels, out_channels, 2)
        self.conv_kxk = _conv(out_channels, out_channels, config.conv_kernel_size, groups=out_channels)
        self.conv_1x1 = _conv(out_channels, width, normalization=False, activation=False)
        blocks = []
        intermediate = int(config.ffn_multiplier * width // 16) * 16
        for _ in range(depth):
            block = VitEncoderBlock(width, 1)
            block.norm1 = GroupNorm(1, width, eps=config.layer_norm_eps)
            block.norm2 = GroupNorm(1, width, eps=config.layer_norm_eps)
            block.attn = _LinearAttention(width)
            block.mlp = nn.Sequential(
                _conv(width, intermediate, normalization=False, bias=True),
                _conv(intermediate, width, normalization=False, activation=False, bias=True),
            )
            blocks.append(block)
        self.transformer = nn.Sequential(*blocks)
        self.layernorm = GroupNorm(1, width, eps=config.layer_norm_eps)
        self.conv_projection = _conv(width, out_channels, activation=False)

    def forward(self, features):
        features = self.downsampling_layer(features)
        features = self.conv_1x1(self.conv_kxk(features))
        height, width = features.shape[-2:]
        patches = self.layernorm(self.transformer(_unfold(features, self.patch)))
        return self.conv_projection(_fold(patches, height, width, self.patch))


class MobileViTV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        multiplier = config.width_multiplier
        channels = [_make_divisible(max(16, min(64, 32 * multiplier)))]
        channels += [_make_divisible(value * multiplier, 16 if index == 0 else 8)
                     for index, value in enumerate((64, 128, 256, 384, 512))]
        widths = [_make_divisible(value * multiplier) for value in config.base_attn_unit_dims]
        self.conv_stem = _conv(config.num_channels, channels[0], 3, 2)
        self.encoder = nn.ModuleList([
            _mobile_stage(config, channels[0], channels[1], 1, 1),
            _mobile_stage(config, channels[1], channels[2], 2, 2),
            *[_MobileViTV2Layer(config, channels[index + 2], channels[index + 3], width, depth)
              for index, (width, depth) in enumerate(zip(widths, config.n_attn_blocks))],
        ])
        self.pooler = GlobalAvgPool2d()

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This MobileViTV2 coverage model supports inference only")
        hidden_states = self.conv_stem(pixel_values)
        for stage in self.encoder:
            hidden_states = stage(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler(hidden_states)}


def build_from_config(config, device, dtype):
    if config.output_stride != 32 or config.hidden_act not in ("swish", "silu"):
        raise ValueError("This case retains the default stride-32 encoder and HF's swish/SiLU activation")
    return MobileViTV2Model(config).to(device=device, dtype=dtype).eval()


def _state_name(name):
    return _mobile_state(name).replace(".attention.", ".attn.").replace(
        ".ffn.conv1.", ".mlp.0.").replace(".ffn.conv2.", ".mlp.1.")


def load_state_dict_into(model, state_dict, config):
    del config
    model.load_state_dict({_state_name(name): value for name, value in state_dict.items()}, strict=True)
