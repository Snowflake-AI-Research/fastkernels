"""Default I-JEPA inference through the existing full SigLIP vision encoder."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit import _pair
from fastkernels.hf_coverage.models.vit_msn import _check_encoder_config, make_workloads
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L4.pi0 import SigLIPVisionConfig, SigLIPVisionEncoder


class IJepaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        height, width = _pair(config.image_size)
        patch_height, patch_width = _pair(config.patch_size)
        if height != width or patch_height != patch_width or config.num_channels != 3:
            raise ValueError("The SigLIP vision encoder requires square images/patches and three channels")
        if not config.qkv_bias:
            raise ValueError("The SigLIP vision encoder requires biased Q/K/V projections")
        self.image_size = height
        self.encoder = SigLIPVisionEncoder(SigLIPVisionConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            image_size=height,
            patch_size=patch_height,
            layer_norm_eps=config.layer_norm_eps,
        ))
        # Select the existing LayerNorm's native-dtype mode to match HF nn.LayerNorm.
        for module in self.encoder.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The I-JEPA coverage model supports inference only")
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError("Expected NCHW pixel_values at the configured image size")
        pixel_values = pixel_values.to(dtype=self.encoder.patch_embedding.weight.dtype)
        return {"last_hidden_state": self.encoder(pixel_values)}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> IJepaModel:
    _check_encoder_config(config)
    return IJepaModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model: IJepaModel, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    remaining = dict(state_dict)
    mapped = {}

    def copy(destination: str, source: str) -> None:
        mapped["encoder." + destination] = remaining.pop(source)

    copy("position_embedding", "embeddings.position_embeddings")
    for field in ("weight", "bias"):
        copy(f"patch_embedding.{field}", f"embeddings.patch_embeddings.projection.{field}")
        copy(f"post_layernorm.{field}", f"layernorm.{field}")
    for index, _ in enumerate(model.encoder.layers):
        for destination, source in (
            ("self_attn.q_proj", "attention.attention.query"),
            ("self_attn.k_proj", "attention.attention.key"),
            ("self_attn.v_proj", "attention.attention.value"),
            ("self_attn.out_proj", "attention.output.dense"),
            ("mlp.fc1", "intermediate.dense"),
            ("mlp.fc2", "output.dense"),
            ("layer_norm1", "layernorm_before"),
            ("layer_norm2", "layernorm_after"),
        ):
            for field in ("weight", "bias"):
                copy(f"layers.{index}.{destination}.{field}", f"encoder.layer.{index}.{source}.{field}")
    if remaining:
        raise KeyError(f"Unmapped I-JEPA state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
