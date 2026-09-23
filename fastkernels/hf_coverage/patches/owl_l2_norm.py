"""OWL's additive denominator epsilon variant of the existing L2Norm."""

import torch

from fastkernels.tasks.baseline.L1.l2_norm import L2Norm


class AdditiveEpsilonL2Norm(L2Norm):
    """Keep the same vector norm and division, adding epsilon instead of clamping.

    The reduction axis, data dependencies and intermediate norm shape match
    L2Norm's F.normalize implementation. Only its scalar denominator adjustment
    changes from max(norm, eps) to norm + eps. This is the executed adapter.
    """

    def forward(self, x):
        return x / (torch.linalg.vector_norm(x, ord=2, dim=self.dim, keepdim=True) + self.eps)
