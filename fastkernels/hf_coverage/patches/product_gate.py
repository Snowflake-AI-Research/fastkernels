"""GeluAndMul CUDA adaptation with identity activation on the gate half."""

import torch
from torch import nn

from fastkernels.infra.cuda_ext import lazy_op

_CUDA = lazy_op("hf_coverage_product_gate", "product_gate.cu")


class ProductGate(nn.Module):
    """Keep the parent's packed pairs, independent outputs, and CUDA storage.

    The included parent kernel executes its existing scalar/vector product
    with an identity activation callback. There is no new reduction or
    communication. CPU execution is a diagnostic arithmetic comparison.
    """

    def forward(self, packed):
        width = packed.shape[-1] // 2
        if packed.shape[-1] != 2 * width:
            raise ValueError("ProductGate expects packed equal-width gate and value halves")
        if not packed.is_cuda:
            return packed[..., :width] * packed[..., width:]
        packed = packed.contiguous()
        output = torch.empty(packed.shape[:-1] + (width,), device=packed.device, dtype=packed.dtype)
        _CUDA.product_gate(output, packed)
        return output
