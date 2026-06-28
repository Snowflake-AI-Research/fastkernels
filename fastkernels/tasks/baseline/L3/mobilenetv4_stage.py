"""MobileNetV4 stage: sequential container for UIB / EdgeResidual / Conv blocks.

Builds a single stage (nn.Sequential) from a list of block config dicts.
Each dict specifies block type and construction args. Supported types:
  - "uib": UniversalInvertedResidual
  - "er":  EdgeResidual (FusedMBConv)
  - "cn":  ConvBlock (Conv + BN + ReLU)

Reference: timm/models/mobilenetv3.py EfficientNetBuilder
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.relu import ReLU
from ..L2.mobilenetv4_edge_residual import EdgeResidual
from ..L2.mobilenetv4_uib import UniversalInvertedResidual


class ConvBlock(nn.Module):
    """Simple Conv + BN + ReLU block (cn_* architecture tokens).

    Submodule naming (.conv, .bn1) matches timm's ConvBnAct in
    _efficientnet_blocks.py for direct state dict compatibility.
    """

    def __init__(self, in_chs: int, out_chs: int, kernel_size: int = 1, stride: int = 1):
        super().__init__()
        self.conv = Conv2d(
            in_chs, out_chs, kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = BatchNorm2d(out_chs)
        self.act = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn1(self.conv(x)))


_BLOCK_BUILDERS = {
    "uib": lambda cfg: UniversalInvertedResidual(
        in_chs=cfg["in_chs"],
        out_chs=cfg["out_chs"],
        dw_kernel_size_start=cfg.get("dw_kernel_size_start", 0),
        dw_kernel_size_mid=cfg.get("dw_kernel_size_mid", 3),
        dw_kernel_size_end=cfg.get("dw_kernel_size_end", 0),
        stride=cfg.get("stride", 1),
        exp_ratio=cfg.get("exp_ratio", 1.0),
    ),
    "er": lambda cfg: EdgeResidual(
        in_chs=cfg["in_chs"],
        out_chs=cfg["out_chs"],
        exp_kernel_size=cfg.get("exp_kernel_size", 3),
        stride=cfg.get("stride", 1),
        exp_ratio=cfg.get("exp_ratio", 1.0),
    ),
    "cn": lambda cfg: ConvBlock(
        in_chs=cfg["in_chs"],
        out_chs=cfg["out_chs"],
        kernel_size=cfg.get("kernel_size", 1),
        stride=cfg.get("stride", 1),
    ),
}


class MobileNetV4Stage(nn.Sequential):
    """A stage containing a sequence of MobileNetV4 blocks.

    Args:
        block_configs: List of dicts, each with a "type" key ("uib", "er",
            or "cn") plus block-specific constructor args.
    """

    def __init__(self, block_configs: List[Dict[str, Any]]):
        blocks = []
        for cfg in block_configs:
            builder = _BLOCK_BUILDERS[cfg["type"]]
            blocks.append(builder(cfg))
        super().__init__(*blocks)
