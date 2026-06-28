"""Offset/batch conversion helpers for point cloud batches."""

from __future__ import annotations

import torch
import torch.nn as nn


class Offset2Bincount(nn.Module):
    def forward(self, offset: torch.Tensor) -> torch.Tensor:
        return torch.diff(
            offset,
            prepend=torch.tensor([0], device=offset.device, dtype=torch.long),
        )


class Offset2Batch(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset2bincount = Offset2Bincount()

    def forward(self, offset: torch.Tensor) -> torch.Tensor:
        bincount = self.offset2bincount(offset)
        return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


class Batch2Offset(nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(batch.bincount(), dim=0).long()
