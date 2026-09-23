"""MLCD's vision encoder with learned CLS rotary angles and normalized CLS pooling."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.hf_coverage.patches.mlcd_rope import LearnedPrefixVisionRotaryEmbedding
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L3.eva_block import EvaBlock


class _Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.empty(config.hidden_size))
        self.patch_embedding = Conv2d(config.num_channels, config.hidden_size,
                                      kernel_size=config.patch_size, stride=config.patch_size, bias=False)

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        patches = patches.flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(pixel_values.shape[0], 1, -1)
        return torch.cat((cls, patches), dim=1)


def _block(config):
    block = EvaBlock(config.hidden_size, config.num_attention_heads,
                     mlp_ratio=config.intermediate_size / config.hidden_size,
                     qkv_bias=True, qkv_fused=False, num_prefix_tokens=0,
                     attn_drop=config.attention_dropout, init_values=None)
    block.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
    block.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
    block.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size,
                              config.hidden_size, act_approximate="none")
    return block


class MLCDVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.embeddings = _Embeddings(config)
        self.pre_layrnorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.encoder = nn.ModuleList([_block(config) for _ in range(config.num_hidden_layers)])
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        rotary_dim = config.hidden_size // config.num_attention_heads // 2
        self.vision_rotary_embedding = LearnedPrefixVisionRotaryEmbedding(rotary_dim)
        self.class_pos_emb = nn.Parameter(torch.empty(1, rotary_dim))

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The MLCD coverage model supports inference only")
        height, width = pixel_values.shape[-2:]
        rope = self.vision_rotary_embedding(height // self.patch_size, width // self.patch_size, self.class_pos_emb)
        hidden_states = self.pre_layrnorm(self.embeddings(pixel_values))
        # FP32 rope promotes EvaAttention's rotation products before its type_as(v) boundary.
        for block in self.encoder:
            hidden_states = block(hidden_states, rope=rope)
        return {"last_hidden_state": hidden_states,
                "pooler_output": self.post_layernorm(hidden_states[:, 0])}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> MLCDVisionModel:
    if config.hidden_act != "gelu" or config.num_key_value_groups != 1:
        raise ValueError("This pilot preserves exact GELU and ordinary multi-head attention")
    if config.num_attention_heads <= 0 or config.hidden_size <= 0 or config.hidden_size % (4 * config.num_attention_heads):
        raise ValueError("Head width must be a positive multiple of four for the two-dimensional RoPE")
    if config.num_hidden_layers <= 0 or config.intermediate_size <= 0 or config.patch_size <= 0:
        raise ValueError("Encoder dimensions and patch size must be positive")
    if getattr(config, "output_hidden_states", False) or getattr(config, "output_attentions", False):
        raise ValueError("This pilot returns the default hidden states and normalized CLS pooler")
    model = MLCDVisionModel(config)
    inv_freq = model.vision_rotary_embedding.inv_freq
    model.to(device=device, dtype=dtype)
    model.vision_rotary_embedding.inv_freq = inv_freq.to(device=device)
    return model.eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace("encoder.layers.", "encoder.")
        name = name.replace(".self_attn.out_proj.", ".attn.proj.").replace(".self_attn.", ".attn.")
        name = name.replace(".layer_norm1.", ".norm1.").replace(".layer_norm2.", ".norm2.")
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
