"""YOLOv10 native backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.yolov10_c2f import YOLOC2f
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_psa import YOLOPSA
from ..L2.yolov10_scdown import YOLOSCDown
from ..L2.yolov10_sppf import YOLOSPPF


class YOLOv10Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = YOLOConv(3, 16, 3, 2)
        self.stem2 = YOLOConv(16, 32, 3, 2)
        self.stage2 = YOLOC2f(32, 32, n=1, shortcut=True)
        self.down3 = YOLOConv(32, 64, 3, 2)
        self.stage3 = YOLOC2f(64, 64, n=2, shortcut=True)
        self.down4 = YOLOSCDown(64, 128, 3, 2)
        self.stage4 = YOLOC2f(128, 128, n=2, shortcut=True)
        self.down5 = YOLOSCDown(128, 256, 3, 2)
        self.stage5 = YOLOC2f(256, 256, n=1, shortcut=True)
        self.sppf = YOLOSPPF(256, 256, 5)
        self.psa = YOLOPSA(256, 256)

    def forward(self, x: torch.Tensor):
        x = self.stem1(x)
        x = self.stem2(x)
        p2 = self.stage2(x)
        x = self.down3(p2)
        p3 = self.stage3(x)
        x = self.down4(p3)
        p4 = self.stage4(x)
        x = self.down5(p4)
        p5 = self.stage5(x)
        p5 = self.sppf(p5)
        p5 = self.psa(p5)
        return {"p3_backbone": p3, "p4_backbone": p4, "p5_backbone": p5}
