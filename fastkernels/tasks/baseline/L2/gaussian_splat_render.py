"""Render block for 3D Gaussian splatting."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gaussian_projection import GaussianProjection
from ..L1.gaussian_rasterize import GaussianRasterize
from ..L1.gaussian_tile_intersection import GaussianTileIntersection


class GaussianSplatRender(nn.Module):
    def __init__(self, tile_size: int = 16):
        super().__init__()
        self.projection = GaussianProjection()
        self.tile_intersection = GaussianTileIntersection(tile_size=tile_size)
        self.rasterize = GaussianRasterize(tile_size=tile_size)

    def forward(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        backgrounds: torch.Tensor | None = None,
        batched_opacities: torch.Tensor | None = None,
        batched_colors: torch.Tensor | None = None,
        return_meta: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        radii, means2d, depths, conics = self.projection(
            means, quats, scales, opacities, viewmats, Ks, width, height
        )
        n_images = viewmats.shape[0]
        if batched_colors is not None:
            colors = batched_colors
        elif colors.dim() == 2:
            colors = colors.expand(n_images, -1, -1)
        if batched_opacities is not None:
            opacities = batched_opacities
        elif opacities.dim() == 1:
            opacities = opacities.expand(n_images, -1)
        isect_offsets, flatten_ids, isect_ids, tile_width, tile_height = self.tile_intersection(
            means2d, radii, depths, width, height, n_images
        )
        render_colors, render_alphas = self.rasterize(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
        )
        if not return_meta:
            return render_colors, render_alphas, {}
        meta = {
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "tile_width": tile_width,
            "tile_height": tile_height,
        }
        return render_colors, render_alphas, meta
