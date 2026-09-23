"""Stable exp-minus-one variant of the existing elementwise Exp operation."""

import torch

from fastkernels.tasks.baseline.L1.tensor_ops import Exp


class Expm1(Exp):
    """Keep Exp's independent elementwise mapping and native PyTorch dispatch.

    Use the expm1 intrinsic to avoid an extra rounded exp output before the
    subtraction. Duration rounding is discontinuous: BF16 x=0.9140625 needs
    expm1(x)=1.4921875, whereas rounded exp(x)-1 gives1.5 and a different count.
    No reduction, communication, state or storage dependency is introduced.
    """

    def forward(self, x):
        return torch.expm1(x)
