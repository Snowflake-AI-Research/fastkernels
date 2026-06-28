"""Interpolate wrapping F.interpolate."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
