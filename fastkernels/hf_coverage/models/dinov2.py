"""Default DINOv2 base-model inference with learned scales and the CLS view."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit import _Embeddings, _pair
from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.eva_block import EvaBlock


def _scaled_block(config, intermediate_size: int, layer_scale: float, *, fused_qkv=True, qkv_bias=True):
    block = EvaBlock(
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        mlp_ratio=intermediate_size / config.hidden_size,
        qkv_bias=qkv_bias,
        qkv_fused=fused_qkv,
        init_values=layer_scale,
        attn_drop=config.attention_probs_dropout_prob,
        proj_drop=config.hidden_dropout_prob,
    )
    block.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
    block.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
    block.mlp = VitEncoderMlp(
        config.hidden_size, intermediate_size, config.hidden_size,
        act_approximate="none", drop=config.hidden_dropout_prob,
    )
    return block


class _Dinov2Embeddings(_Embeddings):
    def __init__(self, config):
        super().__init__(config)
        # The registers variant always owns a mask token; Dinov2Config exposes its presence.
        if getattr(config, "use_mask_token", True):
            self.mask_token = nn.Parameter(torch.empty(1, config.hidden_size))


class Dinov2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = _Dinov2Embeddings(config)
        intermediate = int(config.hidden_size * config.mlp_ratio)
        self.encoder = nn.ModuleList([
            _scaled_block(config, intermediate, config.layerscale_value, qkv_bias=config.qkv_bias)
            for _ in range(config.num_hidden_layers)
        ])
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The DINOv2 coverage models support inference only")
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": hidden_states[:, 0]}


def _check_config(config) -> None:
    if config.hidden_act != "gelu" or config.use_swiglu_ffn:
        raise ValueError("These DINOv2 pilots require the default exact-GELU MLP")
    if config.hidden_size <= 0 or config.mlp_ratio <= 0 or config.num_hidden_layers <= 0:
        raise ValueError("Hidden width, MLP ratio, and encoder depth must be positive")
    if config.num_attention_heads <= 0 or config.hidden_size % config.num_attention_heads:
        raise ValueError("Hidden width must be divisible by a positive attention head count")
    height, width = _pair(config.image_size)
    if height != width or min(height, *_pair(config.patch_size)) <= 0:
        raise ValueError("These pilots use positive square images at the configured patch grid")
    if getattr(config, "output_attentions", False) or getattr(config, "output_hidden_states", False):
        raise ValueError("These pilots return the default final hidden states and CLS pooler view")


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> Dinov2Model:
    _check_config(config)
    return Dinov2Model(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    remaining, mapped = dict(state_dict), {}
    for name in list(remaining):
        if not name.startswith("encoder."):
            destination = name.replace(
                "embeddings.patch_embeddings.projection.", "embeddings.patch_embeddings.proj."
            )
            mapped[destination] = remaining.pop(name)
    for index, block in enumerate(model.encoder):
        source, destination = f"encoder.layer.{index}.", f"encoder.{index}."
        fields = ("weight", "bias") if block.attn.qkv.bias is not None else ("weight",)
        for field in fields:
            mapped[destination + f"attn.qkv.{field}"] = torch.cat([
                remaining.pop(source + f"attention.attention.{name}.{field}")
                for name in ("query", "key", "value")
            ], dim=0)
        for target, origin in (
            ("attn.proj", "attention.output.dense"), ("mlp.fc1", "mlp.fc1"),
            ("mlp.fc2", "mlp.fc2"), ("norm1", "norm1"), ("norm2", "norm2"),
        ):
            for field in ("weight", "bias"):
                mapped[destination + f"{target}.{field}"] = remaining.pop(source + f"{origin}.{field}")
        for number in (1, 2):
            mapped[destination + f"gamma_{number}"] = remaining.pop(source + f"layer_scale{number}.lambda1")
    if remaining:
        raise KeyError(f"Unmapped DINOv2 state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
