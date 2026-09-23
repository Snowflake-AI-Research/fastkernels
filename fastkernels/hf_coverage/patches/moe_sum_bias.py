"""MoeSum adaptation adding JetMoE's learned affine term after the reduction."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.moe_sum import MoeSum


class BiasedMoeSum(MoeSum):
    """Keep the parent's reduction and rounded output; execute the bias add."""

    def __init__(self, width):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(width))

    def forward(self, values, topk):
        return super().forward(values, topk) + self.bias
