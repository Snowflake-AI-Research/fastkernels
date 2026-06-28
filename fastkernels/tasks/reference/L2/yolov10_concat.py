"""YOLOv10 tensor concatenation op."""

from __future__ import annotations

import torch
import torch.nn as nn


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, self.d)
