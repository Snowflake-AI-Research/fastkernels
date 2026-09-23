"""Affine query adaptation of the existing dense-attention operation."""

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention


class BiasedQueryAttention(DenseAttention):
    """Preserve the parent attention and the input-dtype query-bias boundary.

    The learned per-head bias is added before the unchanged attention kernel.
    Folding it into the query projection would change its rounding boundary.
    """

    def forward(self, query, key, value, bias, **kwargs):
        return super().forward(query + bias, key, value, **kwargs)
