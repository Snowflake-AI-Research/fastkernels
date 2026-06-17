"""3D Gaussian projection to 2D image space."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianProjection(nn.Module):
    def __init__(
        self,
        eps2d: float = 0.3,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
    ):
        super().__init__()
        self.eps2d = eps2d
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.radius_clip = radius_clip

    def forward(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from gsplat.rendering import fully_fused_projection

        radii, means2d, depths, conics, _ = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=self.eps2d,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            radius_clip=self.radius_clip,
            packed=False,
            opacities=opacities,
        )
        return radii, means2d, depths, conics
