"""PoolFormer's overlapping embeddings and pooling/channel residual blocks."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.hf_coverage.patches.poolformer_avg_pool import ExcludePaddingAvgPool2d
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.group_norm import GroupNorm
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp


class _Layer(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.before_norm = GroupNorm(1, channels, eps=1e-5)
        self.after_norm = GroupNorm(1, channels, eps=1e-5)
        self.pool = ExcludePaddingAvgPool2d(config.pool_size, stride=1, padding=config.pool_size // 2)
        intermediate = int(channels * config.mlp_ratio)
        self.output = VitEncoderMlp(channels, intermediate, channels, act_approximate="none")
        self.output.fc1 = Conv2d(channels, intermediate, kernel_size=1)
        self.output.fc2 = Conv2d(intermediate, channels, kernel_size=1)
        # A depthwise 1x1 kernel stores one gain per channel, with no channel sum.
        self.layer_scale_1 = Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)
        self.layer_scale_2 = Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)

    def forward(self, hidden_states):
        normalized = self.before_norm(hidden_states)
        hidden_states = hidden_states + self.layer_scale_1(self.pool(normalized) - normalized)
        return hidden_states + self.layer_scale_2(self.output(self.after_norm(hidden_states)))


class PoolFormerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.blocks = nn.ModuleList()
        in_channels = config.num_channels
        for index, channels in enumerate(config.hidden_sizes):
            self.embeddings.append(Conv2d(
                in_channels, channels, config.patch_sizes[index],
                stride=config.strides[index], padding=config.padding[index],
            ))
            self.blocks.append(nn.Sequential(*[_Layer(config, channels) for _ in range(config.depths[index])]))
            in_channels = channels

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The PoolFormer coverage model supports inference only")
        hidden_states = pixel_values
        for embedding, block in zip(self.embeddings, self.blocks):
            hidden_states = block(embedding(hidden_states))
        return {"last_hidden_state": hidden_states}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> PoolFormerModel:
    if config.hidden_act != "gelu" or not config.use_layer_scale:
        raise ValueError("This pilot preserves exact GELU and enabled learned layer scales")
    if config.num_encoder_blocks != 4 or any(len(getattr(config, name)) != 4 for name in
            ("hidden_sizes", "depths", "patch_sizes", "strides", "padding")):
        raise ValueError("This pilot preserves all four PoolFormer stages")
    if any(width <= 0 for width in config.hidden_sizes) or any(depth < 2 for depth in config.depths):
        raise ValueError("Stage widths must be positive and each stage must retain repeated blocks")
    if config.mlp_ratio <= 0 or config.pool_size <= 0 or config.pool_size % 2 != 1:
        raise ValueError("MLP ratio must be positive and the pooling window must be positive and odd")
    if getattr(config, "output_hidden_states", False):
        raise ValueError("This pilot returns the default final feature map")
    return PoolFormerModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace("encoder.patch_embeddings.", "embeddings.").replace(".projection.", ".")
        name = name.replace("encoder.block.", "blocks.")
        name = name.replace(".output.conv1.", ".output.fc1.").replace(".output.conv2.", ".output.fc2.")
        if name.endswith(("layer_scale_1", "layer_scale_2")):
            name += ".weight"
            value = value.reshape(-1, 1, 1, 1)
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
