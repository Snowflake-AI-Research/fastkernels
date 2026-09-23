"""An affine query variant of the existing batched matrix multiplication."""

from fastkernels.tasks.baseline.L1.bmm import BatchMatMul


class BiasedQueryBMM(BatchMatMul):
    """Add the learned query bias before the unchanged BMM reduction.

    Parent: L1 BatchMatMul. Layout, reduction axes, communication and storage
    are unchanged. The separate affine addition preserves the query's dtype
    rounding before multiplication, and its executed cost is measured.
    """

    def forward(self, query, key, bias):
        return super().forward(query + bias, key)
