"""PVT's four positional-embedding stages, spatial reduction and final CLS."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.interpolate import Interpolate
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock

from .pvt_v2 import _EagerAttentionCore
from .segformer import _SpatialAttention
from .vit_msn import make_workloads


class _PatchEmbeddings(OasisPatchEmbed):
    def __init__(self, config, stage):
        image_size = config.image_size if stage == 0 else config.image_size // (2 ** (stage + 1))
        width = config.hidden_sizes[stage]
        super().__init__(image_size, image_size, config.patch_sizes[stage],
                         config.num_channels if stage == 0 else config.hidden_sizes[stage - 1], width,
                         norm_layer=lambda channels: LayerNorm(channels, eps=config.layer_norm_eps,
                                                                promote_fp32=False))
        with_cls = stage == config.num_encoder_blocks - 1
        self.cls_token = nn.Parameter(torch.empty(1, 1, width)) if with_cls else None
        self.position_embeddings = nn.Parameter(torch.empty(1, self.num_patches + int(with_cls), width))
        self.interpolate = Interpolate()

    def forward(self, pixel_values):
        hidden_states = super().forward(pixel_values)
        offset = int(self.cls_token is not None)
        height, width = self.grid_size
        positions = self.position_embeddings[:, offset:].reshape(1, height, width, -1).permute(0, 3, 1, 2)
        # Pinned PVT executes this same-size interpolation at its configured input size.
        positions = self.interpolate(positions, size=(height, width), mode="bilinear")
        positions = positions.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            hidden_states = torch.cat((self.cls_token.expand(hidden_states.shape[0], -1, -1), hidden_states), dim=1)
            positions = torch.cat((self.position_embeddings[:, :1], positions), dim=1)
        return hidden_states + positions


class _PvtAttention(_SpatialAttention):
    def __init__(self, config, stage, grid):
        super().__init__(config.hidden_sizes[stage], config.num_attention_heads[stage],
                         config.sequence_reduction_ratios[stage], config.layer_norm_eps)
        self.grid = grid
        self.attention.attn = _EagerAttentionCore()

    def forward(self, hidden_states, attn_mask=None):
        if attn_mask is not None:
            raise ValueError("The default PVT encoder has no attention mask")
        return super().forward(hidden_states, *self.grid)


class _Stage(nn.Module):
    def __init__(self, config, stage):
        super().__init__()
        self.embedding = _PatchEmbeddings(config, stage)
        self.blocks = nn.ModuleList()
        for _ in range(config.depths[stage]):
            block = VitEncoderBlock(config.hidden_sizes[stage], config.num_attention_heads[stage],
                                    config.mlp_ratios[stage], norm_eps=config.layer_norm_eps)
            block.attn = _PvtAttention(config, stage, self.embedding.grid_size)
            self.blocks.append(block)

    def forward(self, hidden_states):
        hidden_states = self.embedding(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        if self.embedding.cls_token is None:
            height, width = self.embedding.grid_size
            hidden_states = hidden_states.reshape(hidden_states.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()
        return hidden_states


class PvtModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList([_Stage(config, index) for index in range(config.num_encoder_blocks)])
        self.layer_norm = LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, pixel_values):
        if self.training:
            raise RuntimeError("This PVT coverage model supports inference only")
        for stage in self.stages:
            pixel_values = stage(pixel_values)
        return {"last_hidden_state": self.layer_norm(pixel_values)}


def build_from_config(config, device, dtype):
    if config.num_encoder_blocks != 4 or list(config.patch_sizes) != [4, 2, 2, 2] or list(config.strides) != [4, 2, 2, 2]:
        raise ValueError("This case retains PVT's four nonoverlapping patch stages")
    if config.hidden_act != "gelu" or not config.qkv_bias or config.sequence_reduction_ratios[-1] != 1:
        raise ValueError("This case retains exact GELU, biased QKV and unreduced final CLS attention")
    return PvtModel(config).to(device=device, dtype=dtype).eval()


def _state_name(name):
    name = name.removeprefix("encoder.")
    if name.startswith("patch_embeddings."):
        _, stage, rest = name.split(".", 2)
        rest = rest.replace("projection.", "proj.").replace("layer_norm.", "norm.")
        return f"stages.{stage}.embedding.{rest}"
    if name.startswith("block."):
        _, stage, block, rest = name.split(".", 3)
        rest = rest.replace("layer_norm_1.", "norm1.").replace("layer_norm_2.", "norm2.")
        rest = rest.replace("mlp.dense1.", "mlp.fc1.").replace("mlp.dense2.", "mlp.fc2.")
        rest = rest.replace("attention.self.sequence_reduction.", "attn.sr.")
        rest = rest.replace("attention.self.layer_norm.", "attn.norm.")
        for origin, target in (("query", "to_q"), ("key", "to_k"), ("value", "to_v")):
            rest = rest.replace(f"attention.self.{origin}.", f"attn.attention.{target}.")
        rest = rest.replace("attention.output.dense.", "attn.attention.to_out.0.")
        return f"stages.{stage}.blocks.{block}.{rest}"
    return name


def load_state_dict_into(model, state_dict, config):
    del config
    model.load_state_dict({_state_name(name): value for name, value in state_dict.items()}, strict=True)
