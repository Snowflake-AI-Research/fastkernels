"""GELU-and-mul activation for packed gate/up projections.

Counterpart to ``SiluAndMul`` for models using GELU (e.g. Gemma / PaliGemma /
some Pi0 paths).  The input is a concatenation of [gate, up] along the last
dimension; the output is ``gelu(gate) * up`` with the chosen approximation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm._C  # noqa: F401  -- registers torch.ops._C.gelu*_and_mul


class GeluAndMul(nn.Module):
    """Apply GELU to the gate half and multiply by the up half."""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unsupported GELU approximation: {approximate}")
        self.op = (
            torch.ops._C.gelu_tanh_and_mul
            if approximate == "tanh"
            else torch.ops._C.gelu_and_mul
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)
