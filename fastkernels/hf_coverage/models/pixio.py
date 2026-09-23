"""Pixio's eight-CLS-token ViT encoder and default mean-CLS pooler."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit import _Embeddings
from fastkernels.hf_coverage.models.vit_msn import (
    ViTMSNModel, _check_encoder_config, load_state_dict_into as _load_vit, make_workloads,
)
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d


class _PixioEmbeddings(_Embeddings):
    def __init__(self, config):
        super().__init__(config)
        self.cls_token = nn.Parameter(torch.empty(1, config.n_cls_tokens, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.empty(
            1, self.patch_embeddings.num_patches + config.n_cls_tokens, config.hidden_size,
        ))


class PixioModel(ViTMSNModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = _PixioEmbeddings(config)
        self.n_cls_tokens = config.n_cls_tokens
        self.pool = GlobalAvgPool2d()

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(pixel_values)
        cls_tokens = output["last_hidden_state"][:, :self.n_cls_tokens]
        output["pooler_output"] = self.pool(cls_tokens.transpose(1, 2).unsqueeze(-1))
        return output


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> PixioModel:
    values = SimpleNamespace(**config.to_dict())
    values.intermediate_size = int(config.hidden_size * config.mlp_ratio)
    _check_encoder_config(values)
    if config.n_cls_tokens <= 1:
        raise ValueError("This pilot preserves the multiple-CLS prefix and its mean pooler")
    return PixioModel(values).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    canonical = {}
    for name, value in state_dict.items():
        name = name.replace(".norm1.", ".layernorm_before.").replace(".norm2.", ".layernorm_after.")
        name = name.replace(".mlp.fc1.", ".intermediate.dense.").replace(".mlp.fc2.", ".output.dense.")
        canonical[name] = value
    _load_vit(model, canonical, config)
