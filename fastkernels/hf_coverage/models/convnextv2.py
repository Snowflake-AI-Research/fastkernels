"""Default ConvNeXt V2 base-model outputs through the existing complete model."""

from __future__ import annotations

import torch

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L4.convnextv2 import ConvNeXtV2Model as LibraryConvNeXtV2Model


def _check_config(config) -> None:
    if config.num_stages != 4 or len(config.hidden_sizes) != 4 or len(config.depths) != 4:
        raise ValueError("These ConvNeXt pilots preserve all four stages")
    if config.hidden_act != "gelu":
        raise ValueError("These ConvNeXt pilots require the default exact GELU activation")
    if config.num_channels <= 0 or any(width <= 0 for width in config.hidden_sizes):
        raise ValueError("Input and stage channel counts must be positive")
    if any(depth < 2 for depth in config.depths):
        raise ValueError("Each development stage must retain repeated residual blocks")
    if getattr(config, "output_hidden_states", False):
        raise ValueError("These pilots return the default final feature map and pooler output")


class ConvNextV2Model(LibraryConvNeXtV2Model):
    def __init__(self, config):
        super().__init__(config)
        self.num_channels = int(config.num_channels)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The ConvNeXt coverage models support inference only")
        if pixel_values.ndim != 4 or pixel_values.shape[1] != self.num_channels:
            raise ValueError("Expected NCHW pixel_values with the configured channels")
        last_hidden_state, pooled_output = super().forward(pixel_values)
        return {"last_hidden_state": last_hidden_state, "pooler_output": pooled_output}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> ConvNextV2Model:
    _check_config(config)
    return ConvNextV2Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model: ConvNextV2Model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config) -> dict[str, Workload]:
    del config
    if set(inputs) != {"pixel_values"}:
        raise ValueError("Default ConvNeXt inference expects only pixel_values")
    return {"forward": Workload(run=lambda: model(inputs["pixel_values"]))}
