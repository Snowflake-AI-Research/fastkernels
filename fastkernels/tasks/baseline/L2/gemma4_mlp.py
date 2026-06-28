"""Gemma4 dense MLP block."""

from __future__ import annotations

import torch.nn as nn

from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear
from ..L1.gelu_and_mul import GeluAndMul


class Gemma4MLP(nn.Module):
    def __init__(self, config, intermediate_size: int | None = None):
        super().__init__()
        if config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"Unsupported Gemma4 activation: {config.hidden_activation}"
            )
        i = intermediate_size or config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [i, i], bias=False,
        )
        self.down_proj = RowParallelLinear(i, config.hidden_size, bias=False)
        self.act_fn = GeluAndMul("tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))
