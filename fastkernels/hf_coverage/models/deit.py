"""DeiT base-model inference with its CLS and distillation prefix tokens."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit import ViTModel, _Embeddings
from fastkernels.hf_coverage.models.vit_msn import (
    _check_encoder_config,
    load_state_dict_into,
    make_workloads,
)


class _DeiTEmbeddings(_Embeddings):
    def __init__(self, config):
        super().__init__(config)
        self.distillation_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.empty(1, self.patch_embeddings.num_patches + 2, config.hidden_size)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != self.num_channels:
            raise ValueError("Expected NCHW pixel_values with the configured channels")
        pixel_values = pixel_values.to(dtype=self.patch_embeddings.proj.weight.dtype)
        patches = self.patch_embeddings(pixel_values)
        batch_size = pixel_values.shape[0]
        tokens = (
            self.cls_token.expand(batch_size, -1, -1),
            self.distillation_token.expand(batch_size, -1, -1),
            patches,
        )
        return torch.cat(tokens, dim=1) + self.position_embeddings


class DeiTModel(ViTModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = _DeiTEmbeddings(config)


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> DeiTModel:
    _check_encoder_config(config)
    if config.pooler_act != "tanh" or config.pooler_output_size <= 0:
        raise ValueError("Default DeiT pooling requires tanh and a positive output width")
    return DeiTModel(config).to(device=device, dtype=dtype).eval()
