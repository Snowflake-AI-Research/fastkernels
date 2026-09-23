"""L2Norm with ESMFold's separately rounded squared-norm intermediates."""

import torch

from fastkernels.tasks.baseline.L1.l2_norm import L2Norm


class EsmFoldL2Norm(L2Norm):
    """Retain per-vector sum-of-squares reduction and denominator division.

    F.normalize's vector_norm accumulates the norm internally. ESMFold stores
    the squared coordinates, reduced squared norm, and square root in the input
    dtype separately. This adaptation makes those rounding boundaries explicit,
    with the same reduction axis, independent vectors, and linear storage. Its
    epsilon bounds the squared norm, rather than the already computed norm.
    """

    def forward(self, x):
        squared_norm = torch.sum(x.square(), dim=self.dim, keepdim=True)
        denominator = torch.sqrt(squared_norm.clamp(min=self.eps))
        return x / denominator
