"""ConvNeXt V1: the shared four-stage model with existing LayerScale blocks."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.convnextv2 import ConvNextV2Model, _check_config, make_workloads
from fastkernels.tasks.baseline.L1.layer_norm2d import LayerNorm2d
from fastkernels.tasks.baseline.L2.sam3_memory_encoder import CXBlock


def _block(dim: int, layer_scale_init_value: float) -> CXBlock:
    block = CXBlock(
        dim=dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=layer_scale_init_value,
        use_dwconv=True,
    )
    # HF uses native LayerNorm, whereas CXBlock's SAM3 norm computes moments explicitly.
    block.norm = LayerNorm2d(dim, eps=1e-6)
    return block


class ConvNextModel(ConvNextV2Model):
    def __init__(self, config):
        super().__init__(config)
        for stage, dim, depth in zip(self.encoder.stages, config.hidden_sizes, config.depths):
            stage.layers = nn.ModuleList(
                [_block(dim, config.layer_scale_init_value) for _ in range(depth)]
            )


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> ConvNextModel:
    _check_config(config)
    if config.layer_scale_init_value <= 0:
        raise ValueError("This pilot preserves the default-present LayerScale parameters")
    return ConvNextModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model: ConvNextModel, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {}
    for source, value in state_dict.items():
        parts = source.split(".")
        if len(parts) >= 6 and parts[:2] == ["encoder", "stages"] and parts[3] == "layers":
            if parts[5] == "layernorm":
                parts[5] = "norm"
            elif parts[5] == "layer_scale_parameter":
                parts[5] = "gamma"
        destination = ".".join(parts)
        if destination in mapped:
            raise KeyError(f"Multiple ConvNeXt source tensors map to {destination}")
        mapped[destination] = value
    model.load_state_dict(mapped, strict=True)
