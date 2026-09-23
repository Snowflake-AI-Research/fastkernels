"""Fixed affine variants of the existing clipped ReLU pointwise operation."""

import torch
import torch.nn.functional as F
from torch import nn

from .mobilenet_relu6 import ReLU6
from .product_gate import ProductGate


class HardSigmoid(ReLU6):
    """ReLU6(x + 3) / 6, using its fused pointwise implementation.

    The parent clipping interval and independent element mapping are unchanged;
    only fixed input/output affine transforms are added. No reduction, new
    communication or activation-dependent control flow is introduced.
    """
    def forward(self, x):
        return F.hardsigmoid(x)


class HardSwish(nn.Module):
    """Compose the implemented affine-clipping patch and existing product gate."""
    def __init__(self):
        super().__init__()
        self.gate = HardSigmoid()
        self.product = ProductGate()

    def forward(self, x):
        # HF's fused hard-swish stores only the final low-precision result.
        # Keep the composition's intermediate gate in FP32 to match that boundary.
        value = x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
        return self.product(torch.cat((value, self.gate(value)), dim=-1)).to(x.dtype)
