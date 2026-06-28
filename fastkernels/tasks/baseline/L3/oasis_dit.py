"""Oasis diffusion transformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L2.oasis_final_layer import OasisFinalLayer
from ..L2.oasis_patch_embed import OasisPatchEmbed
from ..L2.oasis_timestep_embedder import OasisTimestepEmbedder
from .oasis_block import SpatioTemporalDiTBlock


class OasisDiT(nn.Module):
    def __init__(
        self,
        *,
        input_h: int = 18,
        input_w: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        external_cond_dim: int = 25,
        max_frames: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.x_embedder = OasisPatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = OasisTimestepEmbedder(hidden_size)
        head_dim = hidden_size // num_heads
        self.spatial_rotary_emb = OasisRotaryEmbedding(dim=head_dim // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = OasisRotaryEmbedding(dim=head_dim, freqs_for="lang")
        self.external_cond = Linear(external_cond_dim, hidden_size, bias=True) if external_cond_dim > 0 else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = OasisFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, external_cond: torch.Tensor | None = None) -> torch.Tensor:
        bsz, time, channels, height, width = x.shape
        x = x.reshape(bsz * time, channels, height, width)
        x = self.x_embedder(x)
        x = x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])
        t = t.reshape(bsz * time)
        c = self.t_embedder(t).reshape(bsz, time, -1)
        if torch.is_tensor(external_cond):
            c = c + self.external_cond(external_cond)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = x.reshape(bsz * time, x.shape[2], x.shape[3], x.shape[4])
        x = self.unpatchify(x)
        return x.reshape(bsz, time, x.shape[1], x.shape[2], x.shape[3])
