"""FPN convolution stage for SAM3 neck.

One scale level of the SimpleFPN neck: optional up/downsampling followed by
1x1 and 3x3 convolutions to project to d_model. Each scale factor (4, 2, 1,
0.5) uses a different up/downsample strategy.

Reference: sam3/model/necks.py Sam3DualViTDetNeck conv construction
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.gelu import GELU


class Sam3FPNConvStage(nn.Module):
    """Single FPN scale stage for SAM3 neck.

    Applies scale-dependent up/downsampling then 1x1 + 3x3 conv projection.

    Args:
        in_dim: Input channel dimension from backbone.
        d_model: Output channel dimension.
        scale_factor: Spatial scaling (4.0, 2.0, 1.0, or 0.5).
    """

    def __init__(self, in_dim: int, d_model: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        layers: list[nn.Module] = []

        if scale_factor == 4.0:
            layers.append(nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2))
            layers.append(GELU())
            layers.append(nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2))
            out_dim = in_dim // 4
        elif scale_factor == 2.0:
            layers.append(nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2))
            out_dim = in_dim // 2
        elif scale_factor == 1.0:
            out_dim = in_dim
        elif scale_factor == 0.5:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            out_dim = in_dim
        else:
            raise ValueError(f"Unsupported scale_factor={scale_factor}")

        layers.append(Conv2d(out_dim, d_model, kernel_size=1, bias=True))
        layers.append(Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
