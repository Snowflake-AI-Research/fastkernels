"""Phi4's fixed affine inputs around existing convolution and SiLU gating.

These are input-affine adaptations, not extracted standalone pointwise ops.
The parent convolution reduction and the parent SiLU/product dependency remain
unchanged. Each subtraction, multiplication and bias addition retains its
input-dtype rounding boundary and executes during the measured forward.
"""

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul


class NormalizedAudioConv2d(Conv2d):
    def forward(self, inputs, mean, inverse_std):
        return super().forward(((inputs - mean) * inverse_std).unsqueeze(1))


class BiasedAudioSiluAndMul(SiluAndMul):
    def forward(self, inputs, bias):
        return self.forward_native(inputs + bias)
