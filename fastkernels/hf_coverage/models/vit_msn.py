"""ViT-MSN base-model inference: the ViT encoder with no pooling head."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit import _Embeddings, _encoder_block, _pair
from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm


def _check_encoder_config(config) -> None:
    if config.hidden_act != "gelu":
        raise ValueError("These vision variants require the default exact GELU activation")
    if config.hidden_size <= 0 or config.intermediate_size <= 0:
        raise ValueError("Hidden and intermediate widths must be positive")
    if config.num_attention_heads <= 0 or config.hidden_size % config.num_attention_heads:
        raise ValueError("Hidden width must be divisible by a positive attention head count")
    if config.num_hidden_layers <= 0 or min(*_pair(config.image_size), *_pair(config.patch_size)) <= 0:
        raise ValueError("Encoder depth, image sizes, and patch sizes must be positive")
    if getattr(config, "output_attentions", False) or getattr(config, "output_hidden_states", False):
        raise ValueError("These pilots return the default final model outputs")


class ViTMSNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = _Embeddings(config)
        self.encoder = nn.ModuleList(
            [_encoder_block(config) for _ in range(int(config.num_hidden_layers))]
        )
        self.layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False
        )

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The ViT-MSN coverage model supports inference only")
        hidden_states = self.embeddings(pixel_values)
        for block in self.encoder:
            hidden_states = block(hidden_states)
        return {"last_hidden_state": self.layernorm(hidden_states)}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> ViTMSNModel:
    _check_encoder_config(config)
    return ViTMSNModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    """Map the shared ViT-MSN/DeiT encoder and their explicit embedding/head state."""
    del config
    remaining = dict(state_dict)
    mapped = {}
    for name in list(remaining):
        if not name.startswith("encoder."):
            destination = name.replace(
                "embeddings.patch_embeddings.projection.", "embeddings.patch_embeddings.proj."
            )
            mapped[destination] = remaining.pop(name)
    for index, block in enumerate(model.encoder):
        destination, source = f"encoder.{index}.", f"encoder.layer.{index}."
        fields = ("weight", "bias") if block.attn.qkv.bias is not None else ("weight",)
        for field in fields:
            mapped[destination + f"attn.qkv.{field}"] = torch.cat(
                [remaining.pop(source + f"attention.attention.{name}.{field}")
                 for name in ("query", "key", "value")],
                dim=0,
            )
        for target, origin in (
            ("attn.proj", "attention.output.dense"),
            ("mlp.fc1", "intermediate.dense"),
            ("mlp.fc2", "output.dense"),
            ("norm1", "layernorm_before"),
            ("norm2", "layernorm_after"),
        ):
            for field in ("weight", "bias"):
                mapped[destination + f"{target}.{field}"] = remaining.pop(source + f"{origin}.{field}")
    if remaining:
        raise KeyError(f"Unmapped vision encoder state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config) -> dict[str, Workload]:
    del config
    if set(inputs) != {"pixel_values"}:
        raise ValueError("Default vision inference expects only pixel_values")
    return {"forward": Workload(run=lambda: model(inputs["pixel_values"]))}
