"""HunyuanVideo-1.5 embedding and projection modules (L2 composites).

Contains the patch embedding, adaptive norm, time embedding, ByT5 text
projection, and image projection modules used by the HunyuanVideo-1.5
transformer.

Mirrors the corresponding classes in vllm-omni's
``vllm_omni/diffusion/models/hunyuan_video/hunyuan_video_15_transformer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv3d import Conv3d
from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU
from .timestep_embedding import Timesteps, TimestepEmbedding


class HunyuanVideo15PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class HunyuanVideo15AdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or 2 * in_features
        self.linear = Linear(in_features, out_features)
        self.nonlinearity = SiLU()

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideo15TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, use_meanflow: bool = False):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_meanflow = use_meanflow
        self.time_proj_r = None
        self.timestep_embedder_r = None
        if use_meanflow:
            self.time_proj_r = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder_r = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))

        if timestep_r is not None and self.timestep_embedder_r is not None:
            timesteps_proj_r = self.time_proj_r(timestep_r)
            timesteps_emb_r = self.timestep_embedder_r(timesteps_proj_r.to(dtype=timestep.dtype))
            timesteps_emb = timesteps_emb + timesteps_emb_r

        return timesteps_emb


class HunyuanVideo15ByT5TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.norm = LayerNorm(in_features)
        self.linear_1 = Linear(in_features, hidden_size)
        self.linear_2 = Linear(hidden_size, hidden_size)
        self.linear_3 = Linear(hidden_size, out_features)
        self.act_fn = GELU()

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


class HunyuanVideo15ImageProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.norm_in = LayerNorm(in_channels)
        self.linear_1 = Linear(in_channels, in_channels)
        self.act_fn = GELU()
        self.linear_2 = Linear(in_channels, hidden_size)
        self.norm_out = LayerNorm(hidden_size)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states
