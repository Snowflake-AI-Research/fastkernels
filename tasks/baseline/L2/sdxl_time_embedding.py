"""SDXL time + text_time micro-conditioning embedding (L2 composite).

Reuses Timesteps and TimestepEmbedding from the shared timestep_embedding module.
Adds the SDXL-specific TextTimeEmbedding that combines:
  - pooled text embeddings (from CLIP text_encoder_2)
  - time_ids (crop_coords, target_size) via sinusoidal encoding

Mirrors diffusers' addition_embed_type="text_time" path in UNet2DConditionModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU
from .timestep_embedding import Timesteps, TimestepEmbedding


class TextTimeEmbedding(nn.Module):
    """Projects concatenated (time_ids_sinusoidal + pooled_text_embeds) to time_embed_dim.

    This is the ``add_embedding`` used by SDXL when addition_embed_type="text_time".
    """

    def __init__(self, text_time_input_dim: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = Linear(text_time_input_dim, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x
