"""Thin InstantNGP model wrapper over the L3 pyngp renderer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L3.instantngp_renderer import InstantNGPRenderer


class InstantNGP(nn.Module):
    def __init__(
        self,
        *,
        testbed,
        scene_name: str = "fox",
        width: int,
        height: int,
        spp: int = 1,
        linear: bool = True,
    ):
        super().__init__()
        self.scene_name = scene_name
        self.width = width
        self.height = height
        self.spp = spp
        self.linear = linear
        self.renderer = InstantNGPRenderer(
            testbed=testbed,
            width=width,
            height=height,
            spp=spp,
            linear=linear,
        )

    def render(
        self,
        view_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
        linear: bool | None = None,
    ) -> torch.Tensor:
        return self.renderer.render(
            view_index=view_index,
            width=width,
            height=height,
            spp=spp,
            linear=linear,
        )

    def forward(
        self,
        view_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
    ) -> torch.Tensor:
        return self.render(
            view_index=view_index,
            width=width,
            height=height,
            spp=spp,
        )
