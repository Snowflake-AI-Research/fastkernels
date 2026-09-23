"""DINOv3 ConvNeXt features with a pooled token and final token normalization."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from fastkernels.hf_coverage.models.convnext import ConvNextModel
from fastkernels.hf_coverage.models.convnextv2 import _check_config, make_workloads


class DINOv3ConvNextModel(ConvNextModel):
    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The DINOv3 ConvNeXt coverage model supports inference only")
        features = self.encoder(self.embeddings(pixel_values))
        pooled_token = self.pooler(features).unsqueeze(1)
        tokens = torch.cat((pooled_token, features.flatten(2).transpose(1, 2)), dim=1)
        tokens = self.layernorm(tokens)
        return {"last_hidden_state": tokens, "pooler_output": tokens[:, 0]}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> DINOv3ConvNextModel:
    values = SimpleNamespace(**config.to_dict())
    values.num_stages = len(config.hidden_sizes)
    values.patch_size = 4  # DINOv3's stem has a fixed 4x4 kernel and stride.
    _check_config(values)
    if config.layer_norm_eps != 1e-6 or config.layer_scale_init_value <= 0:
        raise ValueError("This pilot preserves default normalization epsilon and enabled learned scales")
    return DINOv3ConvNextModel(values).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {}
    for name, value in state_dict.items():
        if name.startswith("model.stages.0.downsample_layers."):
            name = name.replace("model.stages.0.downsample_layers.0.", "embeddings.patch_embeddings.")
            name = name.replace("model.stages.0.downsample_layers.1.", "embeddings.layernorm.")
        else:
            name = name.replace("model.stages.", "encoder.stages.")
            name = name.replace(".downsample_layers.", ".downsampling_layer.")
            name = name.replace(".depthwise_conv.", ".dwconv.").replace(".layer_norm.", ".norm.")
            name = name.replace(".pointwise_conv1.", ".pwconv1.").replace(".pointwise_conv2.", ".pwconv2.")
            if name.startswith("layer_norm."):
                name = name.replace("layer_norm.", "layernorm.", 1)
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
