"""BEiT's constructor-default encoder and patch-mean pooling inference."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.dinov2 import _scaled_block
from fastkernels.hf_coverage.models.vit import _pair
from fastkernels.hf_coverage.models.vit_msn import _check_encoder_config, make_workloads
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed


class _BeitEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        height, width = _pair(config.image_size)
        patch_height, patch_width = _pair(config.patch_size)
        if patch_height != patch_width:
            raise ValueError("The patch embedding operation requires square patches")
        self.num_channels = config.num_channels
        self.patch_embeddings = OasisPatchEmbed(
            height, width, patch_height, config.num_channels, config.hidden_size
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != self.num_channels:
            raise ValueError("Expected NCHW pixel_values with the configured channels")
        patches = self.patch_embeddings(pixel_values)
        return torch.cat((self.cls_token.expand(patches.shape[0], -1, -1), patches), dim=1)


class _Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        height, width = _pair(config.image_size)
        patch_height, patch_width = _pair(config.patch_size)
        self.patch_shape = (height // patch_height, width // patch_width)
        self.mean = GlobalAvgPool2d()
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        patches = hidden_states[:, 1:].transpose(1, 2)
        patches = patches.reshape(patches.shape[0], patches.shape[1], *self.patch_shape)
        return self.layernorm(self.mean(patches))


class BeitModel(nn.Module):
    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__()
        self.embeddings = _BeitEmbeddings(config)
        self.encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            block = _scaled_block(
                config, config.intermediate_size, config.layer_scale_init_value, fused_qkv=False
            )
            # BEiT and Data2Vec bias Q and V, while K has no bias parameter.
            block.attn.k_proj = Linear(config.hidden_size, config.hidden_size, bias=False)
            self.encoder.append(block)
        self.pooler = _Pooler(config) if add_pooling_layer else None

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The BEiT/Data2Vec coverage models support inference only")
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        outputs = {"last_hidden_state": hidden_states}
        if self.pooler is not None:
            outputs["pooler_output"] = self.pooler(hidden_states)
        return outputs


def _check_config(config) -> None:
    _check_encoder_config(config)
    if any(getattr(config, name) for name in (
        "use_mask_token", "use_absolute_position_embeddings",
        "use_relative_position_bias", "use_shared_relative_position_bias",
    )):
        raise ValueError("These constructor-default pilots disable masking and position embeddings/bias")
    if not config.use_mean_pooling or config.layer_scale_init_value <= 0:
        raise ValueError("These pilots preserve default mean-pooling configuration and learned layer scales")


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> BeitModel:
    _check_config(config)
    return BeitModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    remaining, mapped = dict(state_dict), {}
    for name in list(remaining):
        if not name.startswith("encoder."):
            destination = name.replace(
                "embeddings.patch_embeddings.projection.", "embeddings.patch_embeddings.proj."
            )
            mapped[destination] = remaining.pop(name)
    for index, _ in enumerate(model.encoder):
        source, destination = f"encoder.layer.{index}.", f"encoder.{index}."
        for target, origin in (
            ("attn.q_proj", "attention.attention.query"),
            ("attn.k_proj", "attention.attention.key"),
            ("attn.v_proj", "attention.attention.value"),
            ("attn.proj", "attention.output.dense"),
            ("mlp.fc1", "intermediate.dense"), ("mlp.fc2", "output.dense"),
            ("norm1", "layernorm_before"), ("norm2", "layernorm_after"),
        ):
            fields = ("weight",) if target == "attn.k_proj" else ("weight", "bias")
            for field in fields:
                mapped[destination + f"{target}.{field}"] = remaining.pop(source + f"{origin}.{field}")
        for number in (1, 2):
            mapped[destination + f"gamma_{number}"] = remaining.pop(source + f"lambda_{number}")
    if remaining:
        raise KeyError(f"Unmapped BEiT/Data2Vec state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
