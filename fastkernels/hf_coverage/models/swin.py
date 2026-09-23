"""Swin windows around existing ViT blocks, with hierarchical patch merging."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.swinv2 import _check_config, make_workloads
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.global_avg_pool2d import GlobalAvgPool2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed
from fastkernels.tasks.baseline.L3.swinv2_block import window_partition, window_reverse
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock


class _WindowBlock(nn.Module):
    def __init__(self, config, stage: int, index: int, resolution: int):
        super().__init__()
        dim, heads = config.embed_dim * 2**stage, config.num_heads[stage]
        self.window = config.window_size
        self.shift = self.window // 2 if index % 2 and resolution > self.window else 0
        self.block = VitEncoderBlock(
            dim, heads, config.mlp_ratio, qkv_bias=config.qkv_bias,
            norm_eps=config.layer_norm_eps, attn_drop=config.attention_probs_dropout_prob,
            proj_drop=config.hidden_dropout_prob,
        )
        self.matmul = BatchMatMul()
        self.softmax = Softmax(dim=-1)
        self.relative_position_bias_table = nn.Parameter(torch.empty((2*self.window-1)**2, heads))
        coordinates = torch.stack(torch.meshgrid(torch.arange(self.window), torch.arange(self.window), indexing="ij"))
        coordinates = coordinates.flatten(1)
        relative = (coordinates[:, :, None] - coordinates[:, None, :]).permute(1, 2, 0)
        relative = relative + self.window - 1
        position_index = relative[..., 0] * (2*self.window-1) + relative[..., 1]
        self.register_buffer("relative_position_index", position_index)
        mask = None
        if self.shift:
            regions = torch.zeros(1, resolution, resolution, 1)
            intervals = (slice(0, -self.window), slice(-self.window, -self.shift), slice(-self.shift, None))
            for h, rows in enumerate(intervals):
                for w, columns in enumerate(intervals):
                    regions[:, rows, columns, :] = 3*h + w
            regions = window_partition(regions, (self.window, self.window)).reshape(-1, self.window**2)
            mask = regions.unsqueeze(1) - regions.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, -100).masked_fill(mask == 0, 0)
        self.register_buffer("shift_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))
        windows = window_partition(x, (self.window, self.window)).reshape(-1, self.window**2, channels)
        attention = self.block.attn
        normalized = self.block.norm1(windows)
        count, tokens, _ = normalized.shape
        heads, head_dim = attention.num_heads, attention.head_dim
        query, key, value = attention.qkv(normalized).reshape(
            count, tokens, 3, heads, head_dim
        ).permute(2, 0, 3, 1, 4).unbind(0)
        # HF stores QK, scaled scores, and probabilities in the model dtype.
        # The existing block's SDPA skips these rounding boundaries. Compose
        # its attention from existing operations while retaining those stores.
        scores = self.matmul(
            query.reshape(count * heads, tokens, head_dim),
            key.reshape(count * heads, tokens, head_dim).transpose(1, 2),
        ).reshape(count, heads, tokens, tokens) / head_dim**0.5
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(self.window**2, self.window**2, -1).permute(2, 0, 1).unsqueeze(0)
        scores = scores + bias
        if self.shift_mask is not None:
            scores = scores + self.shift_mask.repeat(batch, 1, 1).unsqueeze(1)
        context = self.matmul(
            self.softmax(scores).reshape(count * heads, tokens, tokens),
            value.reshape(count * heads, tokens, head_dim),
        ).reshape(count, heads, tokens, head_dim).transpose(1, 2).reshape(count, tokens, channels)
        # Normalization, residuals, and MLPs operate per token, so the whole
        # pre-norm composition can run within each arranged window.
        windows = windows + attention.proj(context)
        windows = windows + self.block.mlp(self.block.norm2(windows))
        windows = windows.reshape(-1, self.window, self.window, channels)
        x = window_reverse(windows, (self.window, self.window), (height, width))
        return torch.roll(x, (self.shift, self.shift), (1, 2)) if self.shift else x


class _PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Swin V1 normalizes before reduction; the existing V2 merge reverses
        # that order and is therefore not the matching composite operation.
        self.norm = LayerNorm(4*dim, promote_fp32=False)
        self.reduction = Linear(4*dim, 2*dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(x))


class _Stage(nn.Module):
    def __init__(self, config, index: int, resolution: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            _WindowBlock(config, index, j, resolution) for j in range(config.depths[index])
        ])
        self.downsample = _PatchMerging(config.embed_dim * 2**index) if index < 3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.downsample(x)


class SwinModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_shape = (config.num_channels, config.image_size, config.image_size)
        self.patch_embed = OasisPatchEmbed(
            config.image_size, config.image_size, config.patch_size,
            in_chans=config.num_channels, embed_dim=config.embed_dim, flatten=False,
            norm_layer=lambda dim: LayerNorm(dim, promote_fp32=False),
        )
        self.stages = nn.ModuleList([
            _Stage(config, i, config.image_size // config.patch_size // 2**i) for i in range(4)
        ])
        self.norm = LayerNorm(config.embed_dim * 8, eps=config.layer_norm_eps, promote_fp32=False)
        self.pool = GlobalAvgPool2d()

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The Swin coverage models support inference only")
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != self.input_shape:
            raise ValueError("pixel_values must match the configured NCHW dimensions")
        x = self.patch_embed(pixel_values)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return {"last_hidden_state": x.reshape(x.shape[0], -1, x.shape[-1]),
                "pooler_output": self.pool(x.permute(0, 3, 1, 2))}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> SwinModel:
    _check_config(config)
    return SwinModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model: SwinModel, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    remaining, mapped = dict(state_dict), {}

    def copy(destination: str, source: str) -> None:
        mapped[destination] = remaining.pop(source)

    norm_mapping = () if isinstance(model.norm, nn.Identity) else (("norm", "layernorm"),)
    for target, source in (("patch_embed.proj", "embeddings.patch_embeddings.projection"),
                           ("patch_embed.norm", "embeddings.norm")) + norm_mapping:
        for field in ("weight", "bias"):
            copy(f"{target}.{field}", f"{source}.{field}")
    for i, stage in enumerate(model.stages):
        if i < 3:
            for field in ("reduction.weight", "norm.weight", "norm.bias"):
                copy(f"stages.{i}.downsample.{field}", f"encoder.layers.{i}.downsample.{field}")
        for j, window in enumerate(stage.blocks):
            target, source = f"stages.{i}.blocks.{j}.", f"encoder.layers.{i}.blocks.{j}."
            for field in ("relative_position_bias_table", "relative_position_index"):
                copy(target + field, source + "attention.self." + field)
            fields = ("weight", "bias") if window.block.attn.qkv.bias is not None else ("weight",)
            for field in fields:
                mapped[target + f"block.attn.qkv.{field}"] = torch.cat([
                    remaining.pop(source + f"attention.self.{name}.{field}") for name in ("query", "key", "value")
                ])
            for dst, src in (("attn.proj", "attention.output.dense"),
                             ("norm1", "layernorm_before"), ("norm2", "layernorm_after"),
                             ("mlp.fc1", "intermediate.dense"), ("mlp.fc2", "output.dense")):
                for field in ("weight", "bias"):
                    copy(target + f"block.{dst}.{field}", source + f"{src}.{field}")
    if remaining:
        raise KeyError(f"Unmapped Swin state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
