"""LLaDA feed-forward network (SwiGLU MLP).

Mirrors the LLaDA/OLMo block MLP: a gated SwiGLU with a separate up-projection.
Submodules are named ``ff_proj`` (gate), ``up_proj`` (up) and ``ff_out`` (down)
so ``infra/llada_loader.py`` maps the checkpoint tensors onto them directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_linear import ReplicatedLinear


class LLaDAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.ff_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=bias)
        self.ff_out = ReplicatedLinear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff_out(F.silu(self.ff_proj(hidden_states)) * self.up_proj(hidden_states))
