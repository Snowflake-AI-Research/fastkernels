"""Data2Vec Vision's default BEiT-style encoder without a pooling layer."""

from __future__ import annotations

import torch

from fastkernels.hf_coverage.models.beit import (
    BeitModel, _check_config, load_state_dict_into, make_workloads,
)


class Data2VecVisionModel(BeitModel):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> Data2VecVisionModel:
    _check_config(config)
    return Data2VecVisionModel(config).to(device=device, dtype=dtype).eval()
