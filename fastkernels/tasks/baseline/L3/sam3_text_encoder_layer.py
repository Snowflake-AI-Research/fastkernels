"""Text transformer layer for SAM3 text encoder.

A single text transformer block wrapping the L2 Sam3TextAttentionBlock.
This L3 module exists for consistency with the layer hierarchy.

Reference: sam3/model/text_encoder_ve.py ResidualAttentionBlock
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..L2.sam3_text_attention import Sam3TextAttentionBlock


class Sam3TextEncoderLayer(nn.Module):
    """Single text transformer layer.

    Thin wrapper around Sam3TextAttentionBlock for L3 consistency.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
    """

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.block = Sam3TextAttentionBlock(d_model, n_head, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.block(x, attn_mask)
