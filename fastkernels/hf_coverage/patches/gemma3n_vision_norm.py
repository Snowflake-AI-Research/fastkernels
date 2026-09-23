"""MobileNetV5's input-dtype precision applied to L1 RMSNormNative.

Retains independent row mean-square reduction and affine multiplication; removes
FP32 promotion and the final cast because timm RMS2d stores each intermediate
in the activation dtype. NCHW/NHWC layout adaptation is outside this operation.
"""
import torch
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative


class Gemma3nVisionNorm(RMSNormNative):
    def forward_native(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)) * self.weight
