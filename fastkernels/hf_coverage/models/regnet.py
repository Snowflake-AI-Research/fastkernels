"""RegNet-Y's grouped bottlenecks, squeeze/excitation, and default pooler."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.efficientnetv2_squeeze_excite import SqueezeExcite
from fastkernels.tasks.baseline.L2.mobilenetv4_uib import ConvNormAct


class _Layer(nn.Module):
    def __init__(self, config, in_channels, out_channels, stride):
        super().__init__()
        self.shortcut = (
            ConvNormAct(in_channels, out_channels, stride=stride, apply_act=False)
            if in_channels != out_channels or stride != 1 else nn.Identity()
        )
        excitation = SqueezeExcite(out_channels, int(round(in_channels / 4)))
        excitation.act1 = ReLU()
        self.layer = nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels, kernel_size=3, stride=stride,
                        groups=max(1, out_channels // config.groups_width)),
            excitation,
            ConvNormAct(out_channels, out_channels, apply_act=False),
        )
        self.activation = ReLU()

    def forward(self, hidden_states):
        return self.activation(self.layer(hidden_states) + self.shortcut(hidden_states))


class RegNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = ConvNormAct(config.num_channels, config.embedding_size, 3, stride=2)
        self.encoder = nn.ModuleList()
        in_channels = config.embedding_size
        for index, (out_channels, depth) in enumerate(zip(config.hidden_sizes, config.depths)):
            stride = 2 if index or config.downsample_in_first_stage else 1
            self.encoder.append(nn.Sequential(
                _Layer(config, in_channels, out_channels, stride),
                *[_Layer(config, out_channels, out_channels, 1) for _ in range(depth - 1)],
            ))
            in_channels = out_channels
        self.pooler = GlobalAvgPool2d(keepdim=True)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The RegNet coverage model supports inference only")
        hidden_states = self.embedder(pixel_values)
        for stage in self.encoder:
            hidden_states = stage(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler(hidden_states)}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> RegNetModel:
    if config.layer_type != "y" or config.hidden_act != "relu":
        raise ValueError("This pilot preserves the default RegNet-Y ReLU graph")
    if len(config.hidden_sizes) != 4 or len(config.depths) != 4 or any(depth < 2 for depth in config.depths):
        raise ValueError("This pilot preserves all four stages and repeated residual blocks")
    if config.groups_width <= 0 or config.num_channels <= 0 or config.embedding_size < 4:
        raise ValueError("Channel and group widths must be positive, with a nonempty excitation bottleneck")
    if any(width < 4 or width % max(1, width // config.groups_width) for width in config.hidden_sizes):
        raise ValueError("Stage widths must allow the configured grouped convolution and excitation")
    if getattr(config, "output_hidden_states", False):
        raise ValueError("This pilot returns the default final feature map and pooler output")
    return RegNetModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace("embedder.embedder.", "embedder.")
        name = name.replace("encoder.stages.", "encoder.").replace(".layers.", ".")
        name = name.replace(".convolution.", ".conv.").replace(".normalization.", ".bn.")
        name = name.replace(".attention.0.", ".conv_reduce.").replace(".attention.2.", ".conv_expand.")
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
