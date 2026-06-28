"""YOLOv10 native neck."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown


class YOLOv10Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)

    def forward(self, feats: dict[str, torch.Tensor]):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        x = self._upsample(p5_backbone, scale_factor=2.0, mode="nearest")
        x = self.cat1([x, p4_backbone])
        p4 = self.c2f_p4(x)

        x = self._upsample(p4, scale_factor=2.0, mode="nearest")
        x = self.cat2([x, p3_backbone])
        p3 = self.c2f_p3(x)

        x = self.down_p3(p3)
        x = self.cat3([x, p4])
        n4 = self.c2f_n4(x)

        x = self.down_n4(n4)
        x = self.cat4([x, p5_backbone])
        n5 = self.c2fcib_n5(x)
        return [p3, n4, n5]
