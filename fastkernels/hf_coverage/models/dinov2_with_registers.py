"""DINOv2 with position-free register tokens inserted after the positioned CLS."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.dinov2 import (
    Dinov2Model, _Dinov2Embeddings, _check_config, load_state_dict_into, make_workloads,
)


class _RegisterEmbeddings(_Dinov2Embeddings):
    def __init__(self, config):
        super().__init__(config)
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(pixel_values)
        registers = self.register_tokens.expand(embeddings.shape[0], -1, -1)
        return torch.cat((embeddings[:, :1], registers, embeddings[:, 1:]), dim=1)


class Dinov2WithRegistersModel(Dinov2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = _RegisterEmbeddings(config)


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> Dinov2WithRegistersModel:
    _check_config(config)
    if config.num_register_tokens <= 0:
        raise ValueError("This pilot preserves the enabled register tokens")
    return Dinov2WithRegistersModel(config).to(device=device, dtype=dtype).eval()
