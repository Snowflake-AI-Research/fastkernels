"""CvT's three convolutional-attention stages and separate final CLS output."""

import math

import torch
from torch import nn

from fastkernels.hf_coverage.runner import config_values
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.mobilenetv4_uib import ConvNormAct
from fastkernels.tasks.baseline.L2.vjepa2_attention import VJEPA2PoolerCrossAttention
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from .vit_msn import make_workloads


class _ConvolutionalAttention(nn.Module):
    def __init__(self, config, stage):
        super().__init__()
        width = config.embed_dim[stage]
        self.with_cls = config.cls_token[stage]
        for name, stride in (("query", config.stride_q[stage]),
                             ("key", config.stride_kv[stage]),
                             ("value", config.stride_kv[stage])):
            setattr(self, name, ConvNormAct(width, width, config.kernel_qkv[stage],
                                           stride=stride, groups=width, apply_act=False))
        self.attention = VJEPA2PoolerCrossAttention(config_values({
            "hidden_size": width, "num_attention_heads": config.num_heads[stage],
        }))
        # Pinned CvT scales by the full embedding width, not the head width.
        self.attention.scale = width ** -0.5
        self.projection = Linear(width, width)

    def forward(self, hidden_states, attn_mask=None):
        batch, tokens, channels = hidden_states.shape
        height = math.isqrt(tokens - int(self.with_cls))
        if attn_mask is not None or height * height != tokens - int(self.with_cls):
            raise ValueError("This CvT case uses square images without attention masks")
        image = hidden_states[:, int(self.with_cls):].transpose(1, 2).reshape(batch, channels, height, height)
        projected = []
        for operation in (self.query, self.key, self.value):
            value = operation(image).flatten(2).transpose(1, 2)
            if self.with_cls:
                value = torch.cat((hidden_states[:, :1], value), dim=1)
            projected.append(value)
        # Selecting this existing eager branch preserves CvT's materialized scores.
        context, _ = self.attention(*projected, output_attentions=True)
        return self.projection(context)


class _Stage(nn.Module):
    def __init__(self, config, stage):
        super().__init__()
        width = config.embed_dim[stage]
        self.projection = Conv2d(config.num_channels if stage == 0 else config.embed_dim[stage - 1],
                                 width, config.patch_sizes[stage], stride=config.patch_stride[stage],
                                 padding=config.patch_padding[stage])
        # CvT constructs plain nn.LayerNorm and does not use config.layer_norm_eps.
        self.normalization = LayerNorm(width, eps=1e-5, promote_fp32=False)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width)) if config.cls_token[stage] else None
        self.layers = nn.ModuleList()
        for _ in range(config.depth[stage]):
            block = VitEncoderBlock(width, config.num_heads[stage], config.mlp_ratio[stage], norm_eps=1e-5)
            block.attn = _ConvolutionalAttention(config, stage)
            self.layers.append(block)

    def forward(self, hidden_states):
        hidden_states = self.projection(hidden_states)
        batch, channels, height, width = hidden_states.shape
        hidden_states = self.normalization(hidden_states.flatten(2).transpose(1, 2))
        if self.cls_token is not None:
            hidden_states = torch.cat((self.cls_token.expand(batch, -1, -1), hidden_states), dim=1)
        for block in self.layers:
            hidden_states = block(hidden_states)
        cls = hidden_states[:, :1] if self.cls_token is not None else None
        hidden_states = hidden_states[:, int(self.cls_token is not None):]
        return hidden_states.transpose(1, 2).reshape(batch, channels, height, width), cls


class CvtModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList([_Stage(config, stage) for stage in range(len(config.depth))])

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This CvT coverage model supports inference only")
        for stage in self.stages:
            pixel_values, cls = stage(pixel_values)
        return {"last_hidden_state": pixel_values, "cls_token_value": cls}


def build_from_config(config, device, dtype):
    if list(config.cls_token) != [False, False, True] or any(method != "dw_bn" for method in config.qkv_projection_method):
        raise ValueError("This case retains CvT's three depthwise-convolution stages and final CLS token")
    if any(not bias for bias in config.qkv_bias) or any(stride != 1 for stride in config.stride_q):
        raise ValueError("This case retains biased QKV and unreduced queries")
    if any(q != kernel // 2 or kv != kernel // 2 for q, kv, kernel in
           zip(config.padding_q, config.padding_kv, config.kernel_qkv)):
        raise ValueError("The default QKV convolution padding is half the kernel size")
    return CvtModel(config).to(device=device, dtype=dtype).eval()


def _state_name(name):
    name = name.removeprefix("encoder.")
    name = name.replace(".embedding.convolution_embeddings.projection.", ".projection.")
    name = name.replace(".embedding.convolution_embeddings.normalization.", ".normalization.")
    for origin, target in (("layernorm_before", "norm1"), ("layernorm_after", "norm2"),
                           ("intermediate.dense", "mlp.fc1"), ("output.dense", "mlp.fc2")):
        if ".attention." not in name:
            name = name.replace(f".{origin}.", f".{target}.")
    for origin, target in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
        name = name.replace(f".attention.attention.projection_{origin}.", f".attn.attention.{target}.")
        name = name.replace(f".attention.attention.convolution_projection_{origin}.convolution_projection.convolution.",
                            f".attn.{origin}.conv.")
        name = name.replace(f".attention.attention.convolution_projection_{origin}.convolution_projection.normalization.",
                            f".attn.{origin}.bn.")
    return name.replace(".attention.output.dense.", ".attn.projection.")


def load_state_dict_into(model, state_dict, config):
    del config
    model.load_state_dict({_state_name(name): value for name, value in state_dict.items()}, strict=True)
