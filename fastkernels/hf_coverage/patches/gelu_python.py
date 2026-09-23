"""GELU adaptation preserving HF's separately rounded Python expression."""

import math

import torch
from torch import nn

from fastkernels.infra.cuda_ext import lazy_op

_CUDA = lazy_op("hf_coverage_gelu_python", "gelu_python.cu")


class PythonGELU(nn.Module):
    """Use the existing activation CUDA kernel with intermediate dtype casts.

    The scalar callback retains the exact GELU formula and independent output
    coordinates. CPU evaluation checks the expression, not the CUDA kernel.
    """

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return hidden_states * 0.5 * (1.0 + torch.erf(hidden_states / math.sqrt(2.0)))
        hidden_states = hidden_states.contiguous()
        output = torch.empty_like(hidden_states)
        _CUDA.gelu_python(output, hidden_states)
        return output
