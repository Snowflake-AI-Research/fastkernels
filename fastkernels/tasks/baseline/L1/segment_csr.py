"""CSR segment reduction via torch.segment_reduce."""

from __future__ import annotations

import torch
import torch.nn as nn


class SegmentCSR(nn.Module):
    def forward(self, src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        reduce_map = {
            "sum": "sum",
            "mean": "mean",
            "min": "amin",
            "max": "amax",
        }
        if reduce not in reduce_map:
            raise ValueError(f"Unsupported reduce op: {reduce}")
        return torch.segment_reduce(src, reduce=reduce_map[reduce], offsets=indptr, axis=0)
