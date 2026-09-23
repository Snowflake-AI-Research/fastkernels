"""ViTDet's documented default global-attention base model."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.hf_coverage.models.vit import _pair
from fastkernels.hf_coverage.models.vit_msn import make_workloads
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L2.rtdetrv2_multihead_attention import RTDetrV2MultiheadAttention
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock


class _Attention(RTDetrV2MultiheadAttention):
    """Select the existing eager attention tensor from its output tuple."""

    def forward(self, hidden_states, attn_mask=None):
        return super().forward(hidden_states, attention_mask=attn_mask)[0]


class VitDetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch = _pair(config.patch_size)
        pretrain_size = _pair(config.pretrain_image_size)
        self.grid_size = (pretrain_size[0] // patch[0], pretrain_size[1] // patch[1])
        self.projection = Conv2d(config.num_channels, config.hidden_size, patch, stride=patch)
        self.position_embeddings = nn.Parameter(torch.empty(1, self.grid_size[0] * self.grid_size[1] + 1, config.hidden_size))
        self.resize = Interpolate()
        self.blocks = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            block = VitEncoderBlock(config.hidden_size, config.num_attention_heads,
                                    mlp_ratio=config.mlp_ratio, norm_eps=config.layer_norm_eps,
                                    proj_drop=config.dropout_prob)
            # HF ViTDet scales Q before its eager QK product.
            block.attn = _Attention(config.hidden_size, config.num_attention_heads)
            self.blocks.append(block)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This coverage model supports inference only")
        hidden_states = self.projection(pixel_values)
        batch_size, channels, height, width = hidden_states.shape
        positions = self.position_embeddings[:, 1:].reshape(1, *self.grid_size, channels)
        if (height, width) != self.grid_size:
            positions = self.resize(positions.permute(0, 3, 1, 2), size=(height, width),
                                    mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        hidden_states = hidden_states.permute(0, 2, 3, 1) + positions
        hidden_states = hidden_states.reshape(batch_size, height * width, channels)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return {"last_hidden_state": hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)}


def build_from_config(config, device, dtype):
    if config.use_relative_position_embeddings or config.window_size or config.window_block_indices or config.residual_block_indices:
        raise ValueError("The documented constructor example uses global attention without relative positions or residual bottlenecks")
    if not config.use_absolute_position_embeddings or not config.qkv_bias or config.hidden_act != "gelu":
        raise ValueError("This pilot preserves absolute positions, biased QKV, and exact GELU")
    if getattr(config, "output_attentions", False) or getattr(config, "output_hidden_states", False):
        raise ValueError("This pilot returns the default final feature map")
    return VitDetModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    mapped = {}
    for name, value in state_dict.items():
        name = name.replace("embeddings.", "").replace("encoder.layer.", "blocks.")
        name = name.replace(".attention.", ".attn.")
        if ".attn.qkv." in name:
            for projection, tensor in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                mapped[name.replace("qkv", projection)] = tensor
        else:
            mapped[name.replace(".attn.proj.", ".attn.out_proj.")] = value
    model.load_state_dict(mapped, strict=True)
