"""Supplied-variance normalization with Moonshine's rounded square-root store.

Parent: forecast_revin.ZeroSafeVarianceNormalize (FrozenBatchNorm2d family).
Keep its independent pointwise supplied-stat mapping; the caller supplies
strictly positive variance+epsilon. Replace reciprocal-square-root multiply by
sqrt, input-dtype store, and divide, matching native frame CMVN rounding.
No new reduction, communication, or enlarged activation layout is introduced.
"""
from torch import nn


class RoundedVarianceNormalize(nn.Module):
    def forward(self, centered, variance):
        return centered / variance.sqrt()
