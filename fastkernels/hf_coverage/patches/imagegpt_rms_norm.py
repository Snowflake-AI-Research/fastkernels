"""ImageGPT's input-dtype precision variant of the existing native RMSNorm."""

import torch
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative


class InputDtypeRMSNorm(RMSNormNative):
    """Preserve the row reduction while rounding each stage to the input dtype.

    Parent: L1 RMSNormNative. ImageGPT rounds the square and mean, then uses
    square-root/division before its affine weight. This is the actual PyTorch
    implementation evaluated and timed, including every launched operation.
    """

    def forward_native(self, inputs):
        variance = inputs.square().mean(dim=-1, keepdim=True)
        normalized = inputs / torch.sqrt(variance + self.eps)
        return normalized * self.weight
