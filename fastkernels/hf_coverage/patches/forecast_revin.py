"""FrozenBatchNorm2d adaptation for the native forecast normalization rule.

The parent maps independent values with broadcast mean/variance and affine
parameters. This variant consumes standard deviation directly, replaces epsilon
stabilization by std<tol -> 1, and preserves the input dtype's subtract store.
It adds no reduction, cross-row communication, or activation-dependent indexing.
"""

import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _normalize(X, Mean, Std, Y, N: tl.constexpr, WIDTH: tl.constexpr,
               TOL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N
    row = index // WIDTH
    x = tl.load(X + index, mask, 0).to(tl.float32)
    mean = tl.load(Mean + row, mask, 0).to(tl.float32)
    std = tl.load(Std + row, mask, 1).to(tl.float32)
    threshold = tl.full((), TOL, tl.float32).to(Std.dtype.element_ty).to(tl.float32)
    safe = tl.where(std < threshold, 1., std)
    centered = (x - mean).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + index, centered / safe, mask)


class ForecastNormalize(nn.Module):
    def __init__(self, tolerance=1e-6):
        super().__init__()
        self.tolerance = tolerance

    def forward(self, values, mean, std):
        if not values.is_cuda:
            safe = torch.where(std < self.tolerance, torch.ones_like(std), std)
            return (values - mean[..., None]) / safe[..., None]
        values, mean, std = values.contiguous(), mean.contiguous(), std.contiguous()
        output = torch.empty_like(values)
        _normalize[(triton.cdiv(values.numel(), 256),)](
            values, mean, std, output, values.numel(), values.shape[-1], self.tolerance, BLOCK=256)
        return output


@triton.jit
def _zero_safe_variance_norm(X, Mean, Variance, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    x = tl.load(X + index, valid, 0).to(tl.float32)
    mean = tl.load(Mean + index, valid, 0).to(tl.float32)
    variance = tl.load(Variance + index, valid, 1).to(tl.float32)
    denominator = tl.where(variance == 0., 1., variance)
    # The approximate reciprocal-square-root instruction flushes subnormal
    # positive variances to zero. Preserve them with rounded sqrt/division.
    tl.store(Y + index, tl.div_rn(x - mean, tl.sqrt_rn(denominator)), valid)


class ZeroSafeVarianceNormalize(nn.Module):
    """FrozenBatchNorm2d without affine, with zero variance regularized to one.

    This is the parent's supplied-stat pointwise mapping with eps=0, replacing
    only its singular zero denominator. The caller composes var/sqrt(var) to
    obtain standard deviation, including exactly zero for constant prefixes.
    """
    def forward(self, values, mean, variance):
        if not values.is_cuda:
            safe = torch.where(variance == 0., torch.ones_like(variance), variance)
            return (values - mean) * safe.rsqrt()
        values, mean, variance = values.contiguous(), mean.contiguous(), variance.contiguous()
        output = torch.empty_like(values)
        _zero_safe_variance_norm[(triton.cdiv(values.numel(), 256),)](
            values, mean, variance, output, values.numel(), BLOCK=256)
        return output
