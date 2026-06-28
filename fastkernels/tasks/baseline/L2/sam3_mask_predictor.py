"""Mask prediction MLP for SAM3 segmentation head.

Maps object query embeddings through a 3-layer MLP, then computes mask logits
via einsum with pixel embeddings from the pixel decoder.

Reference: sam3/model/maskformer_segmentation.py MaskPredictor
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear


class Sam3MaskPredictor(nn.Module):
    """Predicts segmentation masks from object queries and pixel embeddings.

    Args:
        hidden_dim: Dimension of object queries.
        mask_dim: Dimension of pixel embeddings (and MLP output).
    """

    def __init__(self, hidden_dim: int, mask_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(hidden_dim, hidden_dim, bias=True),
            Linear(hidden_dim, hidden_dim, bias=True),
            Linear(hidden_dim, mask_dim, bias=True),
        ])

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    def forward(
        self, obj_queries: torch.Tensor, pixel_embed: torch.Tensor
    ) -> torch.Tensor:
        """Compute mask logits.

        Args:
            obj_queries: (B, Q, D) or (L, B, Q, D) for auxiliary outputs.
            pixel_embed: (B, C, H, W) or (C, H, W) pixel embeddings.

        Returns:
            Mask logits: (B, Q, H, W) or (L, B, Q, H, W).
        """
        mask_embed = self._mlp(obj_queries)

        if mask_embed.ndim == 3:
            if pixel_embed.ndim == 3:
                return torch.einsum("bqc,chw->bqhw", mask_embed, pixel_embed)
            return torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_embed)
        else:
            if pixel_embed.ndim == 3:
                return torch.einsum("lbqc,chw->lbqhw", mask_embed, pixel_embed)
            return torch.einsum("lbqc,bchw->lbqhw", mask_embed, pixel_embed)
