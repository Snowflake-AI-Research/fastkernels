"""Rasterize projected 2D Gaussians to pixels."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianRasterize(nn.Module):
    def __init__(self, tile_size: int = 16):
        super().__init__()
        self.tile_size = tile_size

    def forward(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        width: int,
        height: int,
        isect_offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from gsplat.rendering import rasterize_to_pixels

        return rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            self.tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=False,
        )
