"""Pixel decoder for SAM3 segmentation head.

Progressively upsamples multi-scale FPN features via skip connections,
3x3 conv, group norm, and ReLU. Produces a single-scale pixel embedding
map for mask prediction.

Reference: sam3/model/maskformer_segmentation.py PixelDecoder
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.conv2d import Conv2d


class Sam3PixelDecoder(nn.Module):
    """Pixel decoder that fuses multi-scale features via upsampling.

    Args:
        hidden_dim: Channel dimension of all feature levels.
        num_upsampling_stages: Number of upsampling stages (typically 3).
    """

    def __init__(self, hidden_dim: int, num_upsampling_stages: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages

        conv_layers = []
        norms = []
        for _ in range(num_upsampling_stages):
            conv_layers.append(Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True))
            norms.append(nn.GroupNorm(8, hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)

    def forward(self, backbone_feats: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features.

        Args:
            backbone_feats: List of (B, C, H_i, W_i) feature maps from coarse
                to fine resolution.

        Returns:
            (B, C, H_finest, W_finest) fused pixel embeddings.
        """
        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]

        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            prev_fpn = bb_feat + F.interpolate(
                prev_fpn, size=bb_feat.shape[-2:], mode="nearest"
            )
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn
