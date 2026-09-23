"""ASTModel: overlapping spectrogram patches, ViT blocks, and two-token pooling."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm

from .vit import _encoder_block
from .vit_msn import load_state_dict_into
from ..runner import Workload


class SpectrogramPatches(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = Conv2d(
            1, config.hidden_size, kernel_size=config.patch_size,
            stride=(config.frequency_stride, config.time_stride),
        )

    def forward(self, input_values):
        spectrogram = input_values.unsqueeze(1).transpose(2, 3)
        return self.proj(spectrogram).flatten(2).transpose(1, 2)


class ASTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = SpectrogramPatches(config)
        frequencies = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        times = (config.max_length - config.patch_size) // config.time_stride + 1
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.empty(1, frequencies * times + 2, config.hidden_size))

    def forward(self, input_values):
        batch = input_values.shape[0]
        tokens = (self.cls_token.expand(batch, -1, -1),
                  self.distillation_token.expand(batch, -1, -1),
                  self.patch_embeddings(input_values))
        return torch.cat(tokens, dim=1) + self.position_embeddings


class ASTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ASTEmbeddings(config)
        self.encoder = nn.ModuleList([_encoder_block(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, input_values):
        hidden = self.embeddings(input_values)
        for block in self.encoder:
            hidden = block(hidden)
        hidden = self.layernorm(hidden)
        return {"last_hidden_state": hidden, "pooler_output": (hidden[:, 0] + hidden[:, 1]) / 2}


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu":
        raise ValueError("AST coverage uses its default exact GELU encoder")
    return ASTModel(config).to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
