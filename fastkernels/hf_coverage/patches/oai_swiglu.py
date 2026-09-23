"""Expose existing FastKernels OAI SwiGLU callbacks without changing arithmetic."""

import torch
from torch import nn
from fastkernels.infra.cuda_ext import lazy_op

_CUDA = lazy_op("hf_coverage_oai_swiglu", "oai_swiglu.cu")


class OaiSwiGLU(nn.Module):
    def __init__(self, interleaved=False):
        super().__init__()
        self.interleaved = interleaved

    def forward(self, hidden):
        hidden = hidden.contiguous()
        output = torch.empty(hidden.shape[:-1] + (hidden.shape[-1] // 2,),
                             dtype=hidden.dtype, device=hidden.device)
        if self.interleaved:
            _CUDA.swigluoai_and_mul(output, hidden, 1.702, 7.0)
        else:
            _CUDA.silu_and_mul_clamp(output, hidden, 7.0, 1.702, 1.0)
        return output
