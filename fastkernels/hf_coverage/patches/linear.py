"""Linear adaptations that retain the existing FastKernels matrix multiply."""

from fastkernels.tasks.baseline.L1.linear import Linear


class PostBiasLinear(Linear):
    """Apply bias after the matrix product has rounded to the output dtype.

    Parent: FastKernels L1 Linear. The same Matmul reduction runs without its
    fused bias; a separate add matches HF DeBERTa's packed Q/V bias ordering.
    """

    def forward(self, inputs):
        return self.matmul(inputs, self.weight) + self.bias
