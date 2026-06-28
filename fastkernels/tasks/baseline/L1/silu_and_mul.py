"""SiLU-and-Mul activation with CUDA eager and pure-PyTorch compiled paths."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import vllm._C  # noqa: F401  - registers torch.ops._C.silu_and_mul
    _silu_and_mul_kernel = torch.ops._C.silu_and_mul
except (AttributeError, ImportError):
    from .csrc import _C
    _silu_and_mul_kernel = _C.silu_and_mul


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation — visible to Inductor for fusion."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1) // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        _silu_and_mul_kernel(out, x)
        return out

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)
