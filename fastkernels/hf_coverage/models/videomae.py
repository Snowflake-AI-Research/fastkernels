"""VideoMAE's default unmasked base encoder with complete video tubelets."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.vjepa2_embeddings import VJEPA2PatchEmbeddings3D

from ..runner import Workload
from .vit import _encoder_block


class VideoTubelets(nn.Module):
    def __init__(self, config, tubelet):
        super().__init__()
        temporal, height, width = map(int, tubelet)
        if height != width or min(temporal, height) <= 0:
            raise ValueError("Video coverage uses positive square spatial tubelets")
        self.input_shape = (config.num_frames, config.num_channels, config.image_size, config.image_size)
        self.num_patches = (config.num_frames // temporal) * (config.image_size // height) ** 2
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(
            SimpleNamespace(in_chans=config.num_channels, patch_size=height, tubelet_size=temporal),
            hidden_size=config.hidden_size,
        )

    def forward(self, pixel_values):
        if tuple(pixel_values.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected video shape [batch, {self.input_shape}]")
        return self.patch_embeddings(pixel_values.permute(0, 2, 1, 3, 4))


class VideoMAEEmbeddings(VideoTubelets):
    def __init__(self, config):
        super().__init__(config, (config.tubelet_size, config.patch_size, config.patch_size))
        # Fixed position metadata: match the reference's float64 NumPy table
        # followed by its FP32 storage, before any inference dtype conversion.
        table = np.array([
            [position / np.power(10000, 2 * (index // 2) / config.hidden_size)
             for index in range(config.hidden_size)]
            for position in range(self.num_patches)
        ])
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        self.register_buffer("position_embeddings", torch.tensor(table, dtype=torch.float32).unsqueeze(0), persistent=False)

    def forward(self, pixel_values):
        return super().forward(pixel_values) + self.position_embeddings


class VideoMAEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VideoMAEEmbeddings(config)
        self.encoder = nn.ModuleList([_encoder_block(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = None if config.use_mean_pooling else LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False
        )

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        for block in self.encoder:
            hidden_states = block(hidden_states)
        if self.layernorm is not None:
            hidden_states = self.layernorm(hidden_states)
        return {"last_hidden_state": hidden_states}


def check_video_config(config):
    if config.output_attentions or config.output_hidden_states:
        raise ValueError("Video coverage returns the ordinary final model outputs")
    if config.hidden_size % config.num_attention_heads or config.num_hidden_layers <= 0:
        raise ValueError("The video encoder requires complete positive attention blocks")
    if not isinstance(config.image_size, int):
        raise ValueError("The selected video configurations use a scalar square image size")


def build_from_config(config, device, dtype):
    check_video_config(config)
    if config.hidden_act != "gelu" or not isinstance(config.patch_size, int):
        raise ValueError("VideoMAE coverage preserves the selected GELU and square patch configuration")
    return VideoMAEModel(config).to(device=device, dtype=dtype).eval()


def map_video_state(model, state_dict):
    remaining, mapped = dict(state_dict), {}
    for destination in model.state_dict():
        source = destination.replace(".patch_embeddings.proj.conv.", ".patch_embeddings.projection.")
        if destination.startswith("encoder."):
            _, index, suffix = destination.split(".", 2)
            source = f"encoder.layer.{index}." + suffix
            for target, origin in (
                ("attn.proj", "attention.output.dense"),
                ("mlp.fc1", "intermediate.dense"),
                ("mlp.fc2", "output.dense"),
                ("norm1", "layernorm_before"),
                ("norm2", "layernorm_after"),
            ):
                source = source.replace(target + ".", origin + ".")
            if suffix.startswith("attn.qkv."):
                field = suffix.removeprefix("attn.qkv.")
                mapped[destination] = torch.cat([
                    remaining.pop(f"encoder.layer.{index}.attention.attention.{projection}.{field}")
                    for projection in ("query", "key", "value")
                ])
                continue
        mapped[destination] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped video state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def load_state_dict_into(model, state_dict, config):
    map_video_state(model, state_dict)


def make_workloads(model, inputs, config):
    if set(inputs) != {"pixel_values"}:
        raise ValueError("Default video inference expects only pixel_values")
    return {"forward": Workload(run=lambda: model(inputs["pixel_values"]))}
