"""Functional projection for MobileBERT's factorized, tied MLM weight."""

from fastkernels.tasks.baseline.L1.linear import Matmul


class PostBiasMatmul(Matmul):
    """Retain L1 Matmul's reduction, then add bias after output rounding.

    Unlike PostBiasLinear's stored weight, this interface accepts the weight
    assembled from MobileBERT's two learned factors during each forward call.
    """

    def forward(self, inputs, weight, bias):
        return super().forward(inputs, weight) + bias
