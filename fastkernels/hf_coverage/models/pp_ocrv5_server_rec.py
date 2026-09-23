"""PP-OCRv5 recognition: native backbone, SVTR image sequence, complete CTC scores."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock
from .hgnet_v2 import build_from_config as build_backbone


class Conv(nn.Module):
    def __init__(self, source, target, kernel):
        super().__init__()
        self.convolution = Conv2d(source, target, kernel, padding=tuple(k // 2 for k in kernel), bias=False)
        self.normalization, self.activation = BatchNorm2d(target), SiLU()

    def forward(self, hidden):
        return self.activation(self.normalization(self.convolution(hidden)))


class SVTR(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        width, kernel = config.hidden_size, config.conv_kernel_size
        self.conv_block = nn.ModuleList([
            Conv(channels, channels // 8, kernel), Conv(channels // 8, width, (1, 1)),
            Conv(width, channels, (1, 1)), Conv(2 * channels, channels // 8, kernel),
            Conv(channels // 8, width, (1, 1)),
        ])
        self.svtr_block = nn.ModuleList([
            VitEncoderBlock(width, config.num_attention_heads, mlp_ratio=config.mlp_ratio,
                            qkv_bias=config.qkv_bias, norm_eps=config.layer_norm_eps)
            for _ in range(config.depth)
        ])
        for block in self.svtr_block:
            block.mlp.act = SiLU()
        self.norm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, hidden):
        residual = hidden
        hidden = self.conv_block[1](self.conv_block[0](hidden))
        batch, channels, height, width = hidden.shape
        hidden = hidden.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            hidden = block(hidden)
        hidden = self.norm(hidden).reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        hidden = self.conv_block[2](hidden)
        hidden = self.conv_block[3](torch.cat((residual, hidden), dim=1))
        return self.conv_block[4](hidden).squeeze(2).transpose(1, 2)


class Recognition(nn.Module):
    def __init__(self, config, backbone, channels):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = backbone
        self.head = nn.Module()
        self.head.encoder, self.head.head = SVTR(config, channels), Linear(config.hidden_size, config.head_out_channels)
        self.pool, self.softmax = AvgPool2d((3, 2)), Softmax()

    def forward(self, pixel_values):
        features = self.model.backbone(pixel_values)
        hidden = self.pool(features[f"feature_maps.{len(features) - 1}"])
        logits = self.head.head(self.head.encoder(hidden))
        return {"last_hidden_state": self.softmax(logits.float()).to(logits.dtype)}


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu":
        raise ValueError("The selected OCR recognition configuration uses SiLU")
    backbone = build_backbone(config.backbone_config, device, dtype)
    return Recognition(config, backbone, config.backbone_config.stage_out_channels[-1]).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for destination in model.state_dict():
        source = destination
        if ".svtr_block." in source:
            for old, new in ((".norm1.", ".layer_norm1."), (".norm2.", ".layer_norm2."),
                             (".attn.qkv.", ".self_attn.qkv."), (".attn.proj.", ".self_attn.projection.")):
                source = source.replace(old, new)
        mapped[destination] = remaining.pop(source)
    if remaining:
        raise ValueError(f"Unmapped OCR recognition weights: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
