"""FLUX feed-forward network (L2 composite).

Two-layer MLP: ColumnParallelLinear + GELU(tanh) -> RowParallelLinear.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


class ColumnParallelApproxGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, quant_config=quant_config)
        self.gelu = GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.gelu(x)


class FeedForward(nn.Module):
    """FLUX FFN: GELU(tanh) linear -> linear with TP sharding."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: dict | None = None,
    ) -> None:
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias,
                                      quant_config=quant_config),
            nn.Identity(),
            RowParallelLinear(inner_dim, dim_out, bias=bias, quant_config=quant_config),
        ]
        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
