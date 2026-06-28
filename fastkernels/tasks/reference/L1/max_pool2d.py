"""MaxPool2d wrapping F.max_pool2d."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
        )
