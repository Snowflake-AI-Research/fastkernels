"""Divided temporal/spatial attention with existing attention, MLP and pooling."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.vit_encoder_attention import VitEncoderAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from ..runner import Workload


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frames = config.num_frames
        width = config.hidden_size
        self.temporal_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.layernorm_before = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.layernorm_after = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.temporal_attention = VitEncoderAttention(width, config.num_attention_heads, qkv_bias=config.qkv_bias)
        self.attention = VitEncoderAttention(width, config.num_attention_heads, qkv_bias=config.qkv_bias)
        self.temporal_dense = Linear(width, width)
        self.mlp = VitEncoderMlp(width, config.intermediate_size)
        self.pool = GlobalAvgPool2d()

    def forward(self, hidden):
        batch, _, width = hidden.shape
        frames = self.frames
        patches = (hidden.shape[1] - 1) // frames
        temporal = hidden[:, 1:].reshape(batch * patches, frames, width)
        delta = self.temporal_attention(self.temporal_layernorm(temporal))
        temporal = hidden[:, 1:] + self.temporal_dense(delta.reshape(batch, patches * frames, width))
        cls = hidden[:, :1]
        spatial = temporal.reshape(batch, patches, frames, width).transpose(1, 2).reshape(batch * frames, patches, width)
        spatial = torch.cat((cls.expand(batch, frames, width).reshape(batch * frames, 1, width), spatial), dim=1)
        delta = self.attention(self.layernorm_before(spatial))
        cls_delta = self.pool(delta[:, 0].reshape(batch, frames, width).transpose(1, 2).unsqueeze(-1)).unsqueeze(1)
        patch_delta = delta[:, 1:].reshape(batch, frames, patches, width).transpose(1, 2).reshape(batch, patches * frames, width)
        hidden = torch.cat((cls, temporal), dim=1) + torch.cat((cls_delta, patch_delta), dim=1)
        return hidden + self.mlp(self.layernorm_after(hidden))


class TimesformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frames, self.image_size = config.num_frames, config.image_size
        patches = (config.image_size // config.patch_size) ** 2
        self.projection = Conv2d(config.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.empty(1, patches + 1, config.hidden_size))
        self.time_embeddings = nn.Parameter(torch.empty(1, config.num_frames, config.hidden_size))
        self.layers = nn.ModuleList([_Layer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values):
        batch, frames, channels, height, width = pixel_values.shape
        if frames != self.frames or (height, width) != (self.image_size, self.image_size):
            raise ValueError("TimeSformer inputs must match configured spatial and temporal dimensions")
        hidden = self.projection(pixel_values.reshape(batch * frames, channels, height, width)).flatten(2).transpose(1, 2)
        hidden = torch.cat((self.cls_token.expand(batch * frames, -1, -1), hidden), dim=1) + self.position_embeddings
        cls = hidden[:batch, :1]
        patches, width = hidden.shape[1] - 1, hidden.shape[2]
        hidden = hidden[:, 1:].reshape(batch, frames, patches, width).transpose(1, 2).reshape(batch * patches, frames, width)
        hidden = (hidden + self.time_embeddings).reshape(batch, patches * frames, width)
        hidden = torch.cat((cls, hidden), dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        return {"last_hidden_state": self.layernorm(hidden)}


def build_from_config(config, device, dtype):
    if config.attention_type != "divided_space_time" or config.hidden_act != "gelu":
        raise ValueError("Preserve default divided temporal/spatial attention and exact GELU")
    return TimesformerModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name
        if name in ("cls_token", "position_embeddings", "time_embeddings"):
            source = "embeddings." + name
        elif name.startswith("projection."):
            source = name.replace("projection.", "embeddings.patch_embeddings.projection.")
        elif name.startswith("layers."):
            source = name.replace("layers.", "encoder.layer.", 1)
            source = source.replace("attention.qkv.", "attention.attention.qkv.").replace("attention.proj.", "attention.output.dense.")
            source = source.replace("mlp.fc1.", "intermediate.dense.").replace("mlp.fc2.", "output.dense.")
        mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped TimeSformer state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
