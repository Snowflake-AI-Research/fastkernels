"""Thin InstantNGP renderer wrapper over the official pyngp testbed."""

from __future__ import annotations

import torch
import torch.nn as nn


class InstantNGPRenderer(nn.Module):
    def __init__(
        self,
        *,
        testbed,
        width: int,
        height: int,
        spp: int = 1,
        linear: bool = True,
    ):
        super().__init__()
        self.testbed = testbed
        self.width = width
        self.height = height
        self.spp = spp
        self.linear = linear

    def render(
        self,
        view_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
        linear: bool | None = None,
    ) -> torch.Tensor:
        self.testbed.set_camera_to_training_view(view_index)
        image = self.testbed.render(
            width or self.width,
            height or self.height,
            spp or self.spp,
            self.linear if linear is None else linear,
        )
        return torch.from_numpy(image.copy())

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
