"""ViViT's full spatiotemporal encoder, CLS token, and default pooler."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm

from ..patches.gelu_fast import FastGELU
from .layoutlm import Pooler
from .videomae import VideoTubelets, check_video_config, make_workloads, map_video_state
from .vit import _encoder_block


class VivitEmbeddings(VideoTubelets):
    def __init__(self, config):
        super().__init__(config, config.tubelet_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.empty(1, self.num_patches + 1, config.hidden_size))

    def forward(self, pixel_values):
        patches = super().forward(pixel_values)
        cls_tokens = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        return torch.cat((cls_tokens, patches), dim=1) + self.position_embeddings


class VivitModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VivitEmbeddings(config)
        self.encoder = nn.ModuleList([_encoder_block(config) for _ in range(config.num_hidden_layers)])
        for block in self.encoder:
            block.mlp.act = FastGELU()
        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.pooler = Pooler(config)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        for block in self.encoder:
            hidden_states = block(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler(hidden_states)}


def build_from_config(config, device, dtype):
    check_video_config(config)
    if config.hidden_act != "gelu_fast":
        raise ValueError("ViViT coverage preserves the selected FastGELU activation")
    return VivitModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    # HF calls its Conv3d module tubelet_embeddings rather than patch_embeddings.
    map_video_state(model, {
        key.replace(".tubelet_embeddings.", ".patch_embeddings."): value
        for key, value in state_dict.items()
    })
