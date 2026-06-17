"""TTT-E2E SwiGLU MLP (L2).

Mirrors the reference at ``ttt/model/transformer.py:SwiGLUMLP`` exactly:

    z1 = x @ w1               # (..., intermediate)
    z3 = x @ w3               # (..., intermediate)
    out = (silu(z1) * z3) @ w2

Naming intentionally follows the JAX reference (``w1`` / ``w2`` / ``w3``)
rather than Llama (``gate`` / ``up`` / ``down``) so the state-dict mapping
to and from the reference is direct.

The same module is used for the regular per-block FFN AND the per-suffix-block
"prime" FFN whose weights get test-time-trained chunk-by-chunk.
"""

from __future__ import annotations

import torch
from torch import nn

from ..L1.linear import Linear
from ..L1.silu import SiLU


class TTTE2ESwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))
