"""EfficientNetV2 stage."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.efficientnetv2_edge_residual import EdgeResidual
from ..L2.efficientnetv2_inverted_residual import InvertedResidual


class EfficientNetV2Stage(nn.Module):
    def __init__(self, block_specs: list[dict]):
        super().__init__()
        blocks = []
        for spec in block_specs:
            if spec["kind"] == "edge":
                block = EdgeResidual(
                    in_chs=spec["in_chs"],
                    exp_chs=spec["exp_chs"],
                    out_chs=spec["out_chs"],
                    stride=spec["stride"],
                    has_skip=spec["has_skip"],
                )
            else:
                block = InvertedResidual(
                    in_chs=spec["in_chs"],
                    exp_chs=spec["exp_chs"],
                    out_chs=spec["out_chs"],
                    stride=spec["stride"],
                    se_reduce_chs=spec["se_reduce_chs"],
                    has_skip=spec["has_skip"],
                )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
