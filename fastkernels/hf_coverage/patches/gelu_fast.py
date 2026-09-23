"""Expose the library's existing, otherwise unbound CUDA gelu_fast function."""

import torch
from torch import nn

from fastkernels.infra.cuda_ext import lazy_op

_CUDA = lazy_op("hf_coverage_gelu_fast", "gelu_fast.cu")


class FastGELU(nn.Module):
    """Add a callable interface without changing the parent numerical kernel.

    The CPU expression supports construction checks; it does not validate CUDA.
    """

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return hidden_states * 0.5 * (1.0 + torch.tanh(
                0.79788456 * hidden_states * (1.0 + 0.044715 * hidden_states * hidden_states)
            ))
        hidden_states = hidden_states.contiguous()
        output = torch.empty_like(hidden_states)
        _CUDA.gelu_fast(output, hidden_states)
        return output
