"""Pinned Swinv2Model outputs through the existing complete SwinV2 model."""

from __future__ import annotations

import torch

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L4.swinv2 import SwinV2Model as LibrarySwinV2Model


def _check_config(config) -> None:
    if config.hidden_act != "gelu" or config.use_absolute_embeddings:
        raise ValueError("These Swin pilots preserve exact GELU and no absolute embeddings")
    if len(config.depths) != 4 or len(config.num_heads) != 4:
        raise ValueError("These Swin pilots preserve all four stages")
    if not isinstance(config.image_size, int) or not isinstance(config.patch_size, int):
        raise ValueError("These pilots use fixed square images and patches")
    if config.image_size % config.patch_size:
        raise ValueError("The selected image dimensions must not need patch padding")
    resolution = config.image_size // config.patch_size
    for index, (depth, heads) in enumerate(zip(config.depths, config.num_heads)):
        if depth < 2 or heads <= 0 or (config.embed_dim * 2**index) % heads:
            raise ValueError("Each stage needs repeated blocks and a valid attention head count")
        if resolution < config.window_size or resolution % config.window_size:
            raise ValueError("These workloads preserve window-divisible grids without resizing windows")
        if index < 3:
            if resolution % 2:
                raise ValueError("The selected stages must not need patch-merge padding")
            resolution //= 2
    if getattr(config, "output_hidden_states", False) or getattr(config, "output_attentions", False):
        raise ValueError("These pilots return final features and the default pooler output")


class Swinv2Model(LibrarySwinV2Model):
    def __init__(self, config):
        super().__init__(
            patch_size=config.patch_size,
            in_chans=config.num_channels,
            embed_dim=config.embed_dim,
            depths=tuple(config.depths),
            num_heads=tuple(config.num_heads),
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            default_resolution=config.image_size,
            pretrained_window_sizes=tuple(config.pretrained_window_sizes),
        )
        self.input_shape = (config.num_channels, config.image_size, config.image_size)
        # Pinned HF Swinv2SelfAttention adds the same fixed shift mask twice.
        # This changes only configuration-derived mask values, not model data.
        for stage in self.layers:
            for block in stage.blocks:
                if block.attn_mask is not None:
                    block.attn_mask.mul_(2)
                block.norm1.eps = config.layer_norm_eps
                block.norm2.eps = config.layer_norm_eps
        self.norm.eps = config.layer_norm_eps
        for module in self.modules():
            if isinstance(module, LayerNorm):
                module.promote_fp32 = False

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The Swin coverage models support inference only")
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != self.input_shape:
            raise ValueError("pixel_values must match the configured NCHW dimensions")
        hidden = self.forward_features(pixel_values)
        return {
            "last_hidden_state": hidden.reshape(hidden.shape[0], -1, self.num_features),
            "pooler_output": self.forward_head(hidden),
        }


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> Swinv2Model:
    _check_config(config)
    return Swinv2Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model: Swinv2Model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    remaining, mapped = dict(state_dict), {}

    def copy(destination: str, source: str) -> None:
        mapped[destination] = remaining.pop(source)

    for target, source in (("patch_embed.proj", "embeddings.patch_embeddings.projection"),
                           ("patch_embed.norm", "embeddings.norm"), ("norm", "layernorm")):
        for field in ("weight", "bias"):
            copy(f"{target}.{field}", f"{source}.{field}")
    for i, stage in enumerate(model.layers):
        if i:
            for target in ("reduction.weight", "norm.weight", "norm.bias"):
                copy(f"layers.{i}.downsample.{target}", f"encoder.layers.{i-1}.downsample.{target}")
        for j, block in enumerate(stage.blocks):
            target, source = f"layers.{i}.blocks.{j}.", f"encoder.layers.{i}.blocks.{j}."
            mapped[target + "attn.qkv.weight"] = torch.cat([
                remaining.pop(source + f"attention.self.{name}.weight")
                for name in ("query", "key", "value")
            ])
            if block.attn.q_bias is not None:
                copy(target + "attn.q_bias", source + "attention.self.query.bias")
                copy(target + "attn.v_bias", source + "attention.self.value.bias")
            copy(target + "attn.logit_scale", source + "attention.self.logit_scale")
            for field in ("0.weight", "0.bias", "2.weight"):
                copy(target + "attn.cpb_mlp." + field, source + "attention.self.continuous_position_bias_mlp." + field)
            for dst, src in (("attn.proj", "attention.output.dense"),
                             ("norm1", "layernorm_before"), ("norm2", "layernorm_after"),
                             ("mlp.fc1", "intermediate.dense"), ("mlp.fc2", "output.dense")):
                for field in ("weight", "bias"):
                    copy(target + f"{dst}.{field}", source + f"{src}.{field}")
    if remaining:
        raise KeyError(f"Unmapped SwinV2 state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config) -> dict[str, Workload]:
    del config
    if set(inputs) != {"pixel_values"}:
        raise ValueError("Default Swin inference expects only pixel_values")
    return {"forward": Workload(run=lambda: model(inputs["pixel_values"]))}
