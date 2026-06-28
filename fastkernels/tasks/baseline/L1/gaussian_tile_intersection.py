"""Tile intersection utilities for Gaussian splatting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GaussianTileIntersection(nn.Module):
    def __init__(self, tile_size: int = 16):
        super().__init__()
        self.tile_size = tile_size

    def forward(
        self,
        means2d: torch.Tensor,
        radii: torch.Tensor,
        depths: torch.Tensor,
        width: int,
        height: int,
        n_images: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        from gsplat.rendering import isect_offset_encode, isect_tiles

        tile_width = math.ceil(width / float(self.tile_size))
        tile_height = math.ceil(height / float(self.tile_size))
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            self.tile_size,
            tile_width,
            tile_height,
            packed=False,
            n_images=n_images,
        )
        isect_offsets = isect_offset_encode(
            isect_ids, n_images, tile_width, tile_height
        ).reshape(n_images, tile_height, tile_width)
        return isect_offsets, flatten_ids, isect_ids, tile_width, tile_height
