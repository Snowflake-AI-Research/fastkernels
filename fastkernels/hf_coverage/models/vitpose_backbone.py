"""Default single-expert VitPose backbone through existing ViT blocks."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from ..runner import Workload


class VitPoseBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = tuple(config.image_size)
        patch = tuple(config.patch_size)
        count = (self.image_size[0] // patch[0]) * (self.image_size[1] // patch[1])
        self.projection = Conv2d(config.num_channels, config.hidden_size, patch, stride=patch, padding=2)
        self.position_embeddings = nn.Parameter(torch.empty(1, count + 1, config.hidden_size))
        self.layers = nn.ModuleList([
            VitEncoderBlock(config.hidden_size, config.num_attention_heads, config.mlp_ratio,
                            qkv_bias=config.qkv_bias, norm_eps=config.layer_norm_eps)
            for _ in range(config.num_hidden_layers)
        ])
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.out_indices = tuple(config.out_indices)

    def forward(self, pixel_values):
        if tuple(pixel_values.shape[-2:]) != self.image_size:
            raise ValueError("VitPose inputs must match the configured image dimensions")
        hidden = self.projection(pixel_values).flatten(2).transpose(1, 2)
        hidden = hidden + self.position_embeddings[:, 1:] + self.position_embeddings[:, :1]
        selected = []
        if 0 in self.out_indices:
            selected.append(self.layernorm(hidden))
        for i, block in enumerate(self.layers, 1):
            hidden = block(hidden)
            if i in self.out_indices:
                selected.append(self.layernorm(hidden))
        return {f"feature_maps.{index}": value for index, value in enumerate(selected)}


def build_from_config(config, device, dtype):
    if config.num_experts != 1 or config.hidden_act != "gelu":
        raise ValueError("Preserve the constructor's single-expert GELU path")
    return VitPoseBackbone(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    def copy(dst, src):
        mapped[dst] = remaining.pop(src)
    copy("position_embeddings", "embeddings.position_embeddings")
    for field in ("weight", "bias"):
        copy("projection." + field, "embeddings.patch_embeddings.projection." + field)
        copy("layernorm." + field, "layernorm." + field)
    for i in range(config.num_hidden_layers):
        dst, src = f"layers.{i}.", f"encoder.layer.{i}."
        for field in (("weight", "bias") if config.qkv_bias else ("weight",)):
            mapped[dst + "attn.qkv." + field] = torch.cat([
                remaining.pop(src + f"attention.attention.{name}.{field}")
                for name in ("query", "key", "value")
            ])
        for target, source in (("attn.proj", "attention.output.dense"),
                               ("norm1", "layernorm_before"), ("norm2", "layernorm_after"),
                               ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
            for field in ("weight", "bias"):
                copy(dst + target + "." + field, src + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped VitPose state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
