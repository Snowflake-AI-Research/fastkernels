"""MobileNetV4 Conv Medium image classifier (L4).

Full MobileNetV4 model matching timm's MobileNetV3 container with
head_norm=True for the mobilenetv4_conv_medium variant. 3x3 conv stem,
5 stages (EdgeResidual + UIB + ConvBnAct), efficient head with
post-pool 1x1 conv + BN + ReLU, linear classifier.

Reference: timm/models/mobilenetv3.py MobileNetV3 + _gen_mobilenet_v4
           timm model name: mobilenetv4_conv_medium.e500_r256_in1k
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.global_avg_pool2d import GlobalAvgPool2d
from ..L1.linear import Linear
from ..L1.relu import ReLU
from ..L3.mobilenetv4_stage import MobileNetV4Stage


# Architecture definition for mobilenetv4_conv_medium (channel_multiplier=1.0).
# Each stage is a list of block configs; block indices within each stage match
# timm's EfficientNetBuilder output for state dict key compatibility.
MOBILENETV4_CONV_MEDIUM_STAGES: List[list] = [
    # Stage 0 (112x112 -> 56x56): 1 EdgeResidual
    [
        {"type": "er", "in_chs": 32, "out_chs": 48, "exp_kernel_size": 3, "stride": 2, "exp_ratio": 4},
    ],
    # Stage 1 (56x56 -> 28x28): 2 UIBs
    [
        {"type": "uib", "in_chs": 48, "out_chs": 80, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 5, "stride": 2, "exp_ratio": 4},
        {"type": "uib", "in_chs": 80, "out_chs": 80, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 3, "stride": 1, "exp_ratio": 2},
    ],
    # Stage 2 (28x28 -> 14x14): 8 UIBs
    [
        {"type": "uib", "in_chs": 80,  "out_chs": 160, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 5, "stride": 2, "exp_ratio": 6},
        {"type": "uib", "in_chs": 160, "out_chs": 160, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 3, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 160, "out_chs": 160, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 3, "stride": 1, "exp_ratio": 4},  # r2
        {"type": "uib", "in_chs": 160, "out_chs": 160, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 5, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 160, "out_chs": 160, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 3, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 160, "out_chs": 160, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 160, "out_chs": 160, "dw_kernel_size_start": 0, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 2},
        {"type": "uib", "in_chs": 160, "out_chs": 160, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 4},
    ],
    # Stage 3 (14x14 -> 7x7): 11 UIBs
    [
        {"type": "uib", "in_chs": 160, "out_chs": 256, "dw_kernel_size_start": 5, "dw_kernel_size_mid": 5, "stride": 2, "exp_ratio": 6},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 5, "dw_kernel_size_mid": 5, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 5, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 5, "stride": 1, "exp_ratio": 4},  # r2
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 0, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 3, "dw_kernel_size_mid": 5, "stride": 1, "exp_ratio": 2},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 5, "dw_kernel_size_mid": 5, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 0, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 4},
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 0, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 4},  # r2
        {"type": "uib", "in_chs": 256, "out_chs": 256, "dw_kernel_size_start": 5, "dw_kernel_size_mid": 0, "stride": 1, "exp_ratio": 2},
    ],
    # Stage 4 (7x7): 1 ConvBlock
    [
        {"type": "cn", "in_chs": 256, "out_chs": 960, "kernel_size": 1, "stride": 1},
    ],
]


class MobileNetV4Model(nn.Module):
    """MobileNetV4 Conv Medium image classifier.

    Args:
        stem_size: Stem output channels.
        stages: Per-stage block config lists.
        num_features: Output channels of final block stage.
        head_hidden_size: Channels in the head 1x1 conv.
        num_classes: Classification output dim (0 = feature extractor only).
    """

    def __init__(
        self,
        stem_size: int = 32,
        stages: List[list] | None = None,
        num_features: int = 960,
        head_hidden_size: int = 1280,
        num_classes: int = 1000,
    ):
        super().__init__()
        if stages is None:
            stages = MOBILENETV4_CONV_MEDIUM_STAGES

        self.num_classes = num_classes
        self.num_features = num_features
        self.head_hidden_size = head_hidden_size

        # Stem: 3x3 conv + BN + ReLU
        self.conv_stem = Conv2d(3, stem_size, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(stem_size)
        self.act1 = ReLU()

        # Trunk: sequential stages of blocks
        self.blocks = nn.Sequential(*[
            MobileNetV4Stage(stage_cfgs) for stage_cfgs in stages
        ])

        # Efficient head (MobileNetV4 variant with head_norm)
        self.global_pool = GlobalAvgPool2d(keepdim=True)
        self.conv_head = Conv2d(num_features, head_hidden_size, 1, bias=False)
        self.norm_head = BatchNorm2d(head_hidden_size)
        self.act_head = ReLU()
        self.flatten = nn.Flatten(1)
        self.classifier = Linear(head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.act_head(self.norm_head(self.conv_head(x)))
        x = self.flatten(x)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @staticmethod
    def from_timm(model_name: str = "mobilenetv4_conv_medium.e500_r256_in1k") -> MobileNetV4Model:
        """Load weights from a timm pretrained checkpoint.

        The fastkernels module hierarchy mirrors timm's attribute names so
        state dict keys match directly. The only remapping needed is
        skipping timm-internal buffers that don't exist in our modules.
        """
        import timm

        timm_model = timm.create_model(model_name, pretrained=True)
        timm_sd = timm_model.state_dict()

        num_classes = getattr(timm_model, "num_classes", 1000)
        kb_model = MobileNetV4Model(num_classes=num_classes)

        new_sd = _remap_timm_to_kb(timm_sd)

        missing, unexpected = kb_model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  MobileNetV4 load: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  MobileNetV4 load: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

        del timm_model
        return kb_model


def _remap_timm_to_kb(timm_sd: dict) -> dict:
    """Remap timm MobileNetV3 (V4 variant) state dict to fastkernels MobileNetV4Model.

    Most keys map 1:1 because submodule names match. Keys that don't
    exist in fastkernels (e.g. timm's act2 Identity) are dropped.
    """
    out = {}
    for k, v in timm_sd.items():
        if k.startswith("act2.") or k.startswith("flatten."):
            continue
        out[k] = v
    return out
