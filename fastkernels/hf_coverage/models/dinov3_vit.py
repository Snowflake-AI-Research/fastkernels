"""DINOv3's constructor-default ViT through the existing complete EVA encoder."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.hf_coverage.patches.dinov3_rope import HFDINOv3RoPE
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.eva_attention import EvaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L4.dinov3 import DINOv3Model as LibraryDINOv3Model


class DINOv3ViTModel(LibraryDINOv3Model):
    def __init__(self, config):
        super().__init__(
            patch_size=config.patch_size, in_chans=config.num_channels, embed_dim=config.hidden_size,
            depth=config.num_hidden_layers, num_heads=config.num_attention_heads,
            mlp_ratio=config.intermediate_size / config.hidden_size, init_values=config.layerscale_value,
            rope_temperature=config.rope_theta, num_reg_tokens=config.num_register_tokens,
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.rope = HFDINOv3RoPE(config.hidden_size // config.num_attention_heads, config.rope_theta)
        for block in self.blocks:
            block.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
            block.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
            block.attn = EvaAttention(
                config.hidden_size, config.num_attention_heads, qkv_bias=True, qkv_fused=False,
                num_prefix_tokens=self.num_prefix_tokens, attn_drop=config.attention_dropout,
            )
            for name, bias in (("q_proj", config.query_bias), ("k_proj", config.key_bias),
                               ("v_proj", config.value_bias), ("proj", config.proj_bias)):
                setattr(block.attn, name, Linear(config.hidden_size, config.hidden_size, bias=bias))
            block.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size,
                                      config.hidden_size, bias=config.mlp_bias, act_approximate="none")
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("The DINOv3 ViT coverage model supports inference only")
        pixel_values = pixel_values.to(dtype=self.patch_embed.weight.dtype)
        batch, _, height, width = pixel_values.shape
        hidden_states = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = torch.cat((self.cls_token.expand(batch, -1, -1),
                                   self.reg_token.expand(batch, -1, -1), hidden_states), dim=1)
        rope = self.rope.get_embed([height // self.patch_size, width // self.patch_size]).to(pixel_values.dtype)
        for block in self.blocks:
            hidden_states = block(hidden_states, rope=rope)
        hidden_states = self.norm(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": hidden_states[:, 0]}


def build_from_config(config, device: torch.device, dtype: torch.dtype) -> DINOv3ViTModel:
    if config.hidden_act != "gelu" or config.use_gated_mlp:
        raise ValueError("This pilot preserves the default exact-GELU MLP")
    if config.num_attention_heads <= 0 or config.hidden_size % (4 * config.num_attention_heads):
        raise ValueError("Head width must be a positive multiple of four for the two-dimensional RoPE")
    if config.num_hidden_layers <= 0 or config.intermediate_size <= 0 or config.num_register_tokens < 0:
        raise ValueError("Encoder dimensions must be positive and register count nonnegative")
    if getattr(config, "output_hidden_states", False) or getattr(config, "output_attentions", False):
        raise ValueError("This pilot returns the default hidden states and CLS pooler view")
    model = DINOv3ViTModel(config)
    inv_freq = model.rope.inv_freq
    model.to(device=device, dtype=dtype)
    model.rope.inv_freq = inv_freq.to(device=device)  # HF loading preserves this nonpersistent FP32 buffer.
    return model.eval()


def load_state_dict_into(model, state_dict: dict[str, torch.Tensor], config) -> None:
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace("embeddings.patch_embeddings.", "patch_embed.")
        name = name.replace("embeddings.cls_token", "cls_token").replace("embeddings.mask_token", "mask_token")
        name = name.replace("embeddings.register_tokens", "reg_token")
        name = name.replace("model.layer.", "blocks.").replace(".attention.o_proj.", ".attn.proj.")
        name = name.replace(".attention.", ".attn.")
        name = name.replace(".mlp.up_proj.", ".mlp.fc1.").replace(".mlp.down_proj.", ".mlp.fc2.")
        name = name.replace(".layer_scale1.lambda1", ".gamma_1").replace(".layer_scale2.lambda1", ".gamma_2")
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)
