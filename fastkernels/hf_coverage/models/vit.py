"""Default ViTModel inference: patch tokens, pre-norm encoder, and CLS pooler."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock


def _pair(value) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("Image and patch sizes must be scalars or pairs")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class _Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        height, width = _pair(config.image_size)
        patch_height, patch_width = _pair(config.patch_size)
        if patch_height != patch_width:
            raise ValueError("The OasisPatchEmbed operation requires square patches")
        self.num_channels = int(config.num_channels)
        self.patch_embeddings = OasisPatchEmbed(
            img_height=height,
            img_width=width,
            patch_size=patch_height,
            in_chans=self.num_channels,
            embed_dim=int(config.hidden_size),
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.empty(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != self.num_channels:
            raise ValueError("Expected NCHW pixel_values with the configured channels")
        pixel_values = pixel_values.to(dtype=self.patch_embeddings.proj.weight.dtype)
        patches = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        return torch.cat((cls_tokens, patches), dim=1) + self.position_embeddings


def _encoder_block(config) -> VitEncoderBlock:
    hidden, intermediate = int(config.hidden_size), int(config.intermediate_size)
    block = VitEncoderBlock(
        dim=hidden,
        num_heads=int(config.num_attention_heads),
        mlp_ratio=intermediate / hidden,
        qkv_bias=bool(config.qkv_bias),
        proj_bias=True,
        act_approximate="none",
        attn_drop=float(config.attention_probs_dropout_prob),
        proj_drop=float(config.hidden_dropout_prob),
        norm_eps=float(config.layer_norm_eps),
    )
    # The HF width is an integer; avoid losing a unit to ratio rounding.
    if block.mlp.fc1.weight.shape[0] != intermediate:
        block.mlp = VitEncoderMlp(
            hidden, intermediate, hidden, drop=float(config.hidden_dropout_prob)
        )
    return block


class _Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.pooler_output_size, bias=True)
        self.activation = Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(hidden_states[:, 0]))


class ViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = _Embeddings(config)
        self.encoder = nn.ModuleList(
            [_encoder_block(config) for _ in range(int(config.num_hidden_layers))]
        )
        self.layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False
        )
        self.pooler = _Pooler(config)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The ViT coverage model supports inference only")
        hidden_states = self.embeddings(pixel_values)
        for block in self.encoder:
            hidden_states = block(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": self.pooler(hidden_states),
        }


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> ViTModel:
    if config.hidden_act != "gelu" or config.pooler_act != "tanh":
        raise ValueError("This pilot requires the default GELU encoder and tanh pooler")
    if config.hidden_size <= 0 or config.intermediate_size <= 0 or config.pooler_output_size <= 0:
        raise ValueError("Hidden, intermediate, and pooler widths must be positive")
    if config.num_attention_heads <= 0 or config.hidden_size % config.num_attention_heads:
        raise ValueError("Hidden width must be divisible by a positive attention head count")
    if config.num_hidden_layers <= 0 or min(*_pair(config.image_size), *_pair(config.patch_size)) <= 0:
        raise ValueError("The encoder depth, image sizes, and patch sizes must be positive")
    if getattr(config, "output_attentions", False) or getattr(config, "output_hidden_states", False):
        raise ValueError("This pilot returns the default final hidden states and pooler output")
    return ViTModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model: ViTModel, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    remaining = dict(state_dict)
    mapped = {}

    def copy(destination: str, source: str) -> None:
        mapped[destination] = remaining.pop(source)

    for name in ("cls_token", "position_embeddings"):
        copy(f"embeddings.{name}", f"embeddings.{name}")
    for field in ("weight", "bias"):
        copy(f"embeddings.patch_embeddings.proj.{field}", f"embeddings.patch_embeddings.projection.{field}")
        copy(f"layernorm.{field}", f"layernorm.{field}")
        copy(f"pooler.dense.{field}", f"pooler.dense.{field}")

    for index, block in enumerate(model.encoder):
        destination, source = f"encoder.{index}.", f"encoder.layer.{index}."
        fields = ("weight", "bias") if block.attn.qkv.bias is not None else ("weight",)
        for field in fields:
            mapped[destination + f"attn.qkv.{field}"] = torch.cat(
                [remaining.pop(source + f"attention.attention.{name}.{field}") for name in ("query", "key", "value")],
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
                copy(destination + f"{target}.{field}", source + f"{origin}.{field}")
    if remaining:
        raise KeyError(f"Unmapped ViT state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model: ViTModel, inputs, config) -> dict[str, Workload]:
    del config
    if set(inputs) != {"pixel_values"}:
        raise ValueError("Default ViT inference expects only pixel_values")
    return {"forward": Workload(run=lambda: model(inputs["pixel_values"]))}
