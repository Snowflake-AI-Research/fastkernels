"""YOLOv10 Spatial Pyramid Pooling - Fast."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
