"""Gemma4 precision ordering applied to the existing native RMSNorm reduction."""
import torch
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative


class Gemma4RMSNorm(RMSNormNative):
    """Use pow(-.5), and apply the learned scale before the final dtype cast.

    Parent: RMSNormNative. The row mean-square reduction, dependencies and
    linear storage are unchanged. Gemma4 explicitly specifies pow rather than
    rsqrt and retains FP32 through its affine multiplication.
    """
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__(dim, eps)
        if not with_scale:
            self.register_parameter('weight', None)

    def forward_native(self, x):
        xf = x.float()
        normalized = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -.5)
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        return normalized.to(x.dtype)
