"""SegFormer semantic segmentation, including the complete four-scale decoder."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.batch_norm2d import BatchNorm2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.sdxl_attention import SDXLAttention


def _norm(width, eps):
    return LayerNorm(width, eps=eps, promote_fp32=False)


class _SpatialAttention(nn.Module):
    def __init__(self, width, heads, reduction, eps):
        super().__init__()
        self.reduction = reduction
        if reduction > 1:
            self.sr = Conv2d(width, width, reduction, stride=reduction)
            self.norm = _norm(width, eps)
        self.attention = SDXLAttention(width, heads=heads, dim_head=width // heads)
        # The existing cross-attention carrier accepts independently reduced KV.
        for name in ("to_q", "to_k", "to_v"):
            setattr(self.attention, name, Linear(width, width, bias=True))

    def forward(self, hidden_states, height, width):
        context = hidden_states
        if self.reduction > 1:
            context = context.transpose(1, 2).reshape(context.shape[0], -1, height, width)
            context = self.sr(context).flatten(2).transpose(1, 2)
            context = self.norm(context)
        return self.attention(hidden_states, encoder_hidden_states=context)


class _MixFFN(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.dense1 = Linear(width, intermediate)
        self.dwconv = Conv2d(intermediate, intermediate, 3, padding=1, groups=intermediate)
        self.activation = GELU(approximate="none")
        self.dense2 = Linear(intermediate, width)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1, height, width)
        hidden_states = self.dwconv(hidden_states).flatten(2).transpose(1, 2)
        return self.dense2(self.activation(hidden_states))


class _Block(nn.Module):
    def __init__(self, config, stage, eps):
        super().__init__()
        width = config.hidden_sizes[stage]
        self.norm1 = _norm(width, eps)
        self.attention = _SpatialAttention(width, config.num_attention_heads[stage], config.sr_ratios[stage], eps)
        self.norm2 = _norm(width, eps)
        self.mlp = _MixFFN(width, int(width * config.mlp_ratios[stage]))

    def forward(self, hidden_states, height, width):
        hidden_states = hidden_states + self.attention(self.norm1(hidden_states), height, width)
        return hidden_states + self.mlp(self.norm2(hidden_states), height, width)


class _Stage(nn.Module):
    def __init__(self, config, index, eps):
        super().__init__()
        width = config.hidden_sizes[index]
        kernel = config.patch_sizes[index]
        self.projection = Conv2d(config.num_channels if index == 0 else config.hidden_sizes[index - 1],
                                 width, kernel, stride=config.strides[index], padding=kernel // 2)
        self.embedding_norm = _norm(width, eps)
        self.blocks = nn.ModuleList([_Block(config, index, eps) for _ in range(config.depths[index])])
        self.norm = _norm(width, eps)

    def forward(self, hidden_states):
        hidden_states = self.projection(hidden_states)
        batch_size, _, height, width = hidden_states.shape
        hidden_states = self.embedding_norm(hidden_states.flatten(2).transpose(1, 2))
        for block in self.blocks:
            hidden_states = block(hidden_states, height, width)
        return self.norm(hidden_states).reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()


class SpatialReductionEncoder(nn.Module):
    """The common default SegFormer/PVTv2 encoder graph; epsilon is source-specific."""

    def __init__(self, config, norm_eps):
        super().__init__()
        self.stages = nn.ModuleList([_Stage(config, index, norm_eps) for index in range(config.num_encoder_blocks)])

    def forward(self, pixel_values, all_stages=False):
        features = []
        hidden_states = pixel_values
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            if all_stages:
                features.append(hidden_states)
        return features if all_stages else hidden_states


class _DecodeHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.decoder_hidden_size
        self.linear_c = nn.ModuleList([Linear(channels, width) for channels in config.hidden_sizes])
        self.resize = Interpolate()
        self.linear_fuse = Conv2d(width * config.num_encoder_blocks, width, 1, bias=False)
        self.batch_norm = BatchNorm2d(width)
        self.activation = ReLU()
        self.classifier = Conv2d(width, len(config.id2label), 1)

    def forward(self, features):
        projected = []
        for hidden_states, projection in zip(features, self.linear_c):
            batch_size, _, height, width = hidden_states.shape
            hidden_states = projection(hidden_states.flatten(2).transpose(1, 2))
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, height, width)
            projected.append(self.resize(hidden_states, size=features[0].shape[-2:], mode="bilinear", align_corners=False))
        hidden_states = self.linear_fuse(torch.cat(projected[::-1], dim=1))
        return self.classifier(self.activation(self.batch_norm(hidden_states)))


class SegformerForSemanticSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pinned SegFormer uses nn.LayerNorm's epsilon, ignoring config.layer_norm_eps.
        self.encoder = SpatialReductionEncoder(config, norm_eps=1e-5)
        self.decode_head = _DecodeHead(config)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        return {"logits": self.decode_head(self.encoder(pixel_values, all_stages=True))}


def check_encoder_config(config):
    if config.hidden_act != "gelu" or config.hidden_dropout_prob != 0 or config.attention_probs_dropout_prob != 0:
        raise ValueError("This pilot preserves the default exact GELU and zero attention/hidden dropout")
    if config.num_encoder_blocks != 4 or any(depth < 1 for depth in config.depths):
        raise ValueError("This pilot preserves all four encoder stages")
    if any(width % heads for width, heads in zip(config.hidden_sizes, config.num_attention_heads)):
        raise ValueError("Each stage width must be divisible by its head count")
    if getattr(config, "output_attentions", False) or getattr(config, "output_hidden_states", False):
        raise ValueError("This pilot returns the ordinary task outputs")


def build_from_config(config, device, dtype):
    check_encoder_config(config)
    if not config.reshape_last_stage:
        raise ValueError("The documented checkpoint reshapes every encoder stage to NCHW")
    return SegformerForSemanticSegmentation(config).to(device=device, dtype=dtype).eval()


def encoder_state_name(name):
    """Map a stage's source block names shared by SegFormer and PVTv2."""
    name = name.replace(".layer_norm_1.", ".norm1.").replace(".layer_norm_2.", ".norm2.")
    name = name.replace(".mlp.dwconv.dwconv.", ".mlp.dwconv.")
    name = name.replace(".attention.layer_norm.", ".attention.norm.")
    for origin, target in (("query", "to_q"), ("key", "to_k"), ("value", "to_v"), ("proj", "to_out.0")):
        name = name.replace(f".attention.{origin}.", f".attention.attention.{target}.")
    return name


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for name, value in state_dict.items():
        if name.startswith("segformer.encoder."):
            name = name.removeprefix("segformer.encoder.")
            for index in range(config.num_encoder_blocks):
                name = name.replace(f"patch_embeddings.{index}.proj.", f"encoder.stages.{index}.projection.")
                name = name.replace(f"patch_embeddings.{index}.layer_norm.", f"encoder.stages.{index}.embedding_norm.")
                name = name.replace(f"layer_norm.{index}.", f"encoder.stages.{index}.norm.")
                name = name.replace(f"block.{index}.", f"encoder.stages.{index}.blocks.")
            name = name.replace(".attention.self.", ".attention.").replace(".attention.output.dense.", ".attention.proj.")
            name = encoder_state_name(name)
        else:
            for index in range(config.num_encoder_blocks):
                name = name.replace(f"linear_c.{index}.proj.", f"linear_c.{index}.")
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
