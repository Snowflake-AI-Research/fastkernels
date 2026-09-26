"""DenseAttention's SDPA operation with native grouped-query storage."""

from contextlib import nullcontext

from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention


class GroupedDenseAttention(DenseAttention):
    """Add SDPA's GQA argument instead of materializing repeated key/value heads.

    The parent's dense attention reduction remains unchanged. The default permits
    cuDNN and math; ``backend="sdpa"`` retains PyTorch's ordinary selection.
    Native grouped storage changes rounding relative to repeated heads during
    single-token decode; trained InternVL requires this distinction. Timings
    include whichever permitted backend PyTorch selects.
    """

    def __init__(self, backend="cudnn"):
        if backend not in ("cudnn", "sdpa"):
            raise ValueError("GroupedDenseAttention supports cudnn or sdpa")
        super().__init__(backend=backend)

    def forward(self, query, key, value, causal=False):
        query, key, value = (tensor.permute(0, 2, 1, 3) for tensor in (query, key, value))
        dispatch = (sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH])
                    if self.use_cudnn_kernel else nullcontext())
        with dispatch:
            output = F.scaled_dot_product_attention(query, key, value, is_causal=causal,
                                                     dropout_p=0.0, enable_gqa=True)
        return output.transpose(1, 2)
