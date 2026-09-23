"""Add Falcon-H1's static channel scaling after an existing projection.

Parent: Mamba2Mixer.in_proj (ColumnParallelLinear). The unchanged GEMM emits
the usual z/x/B/C/dt layout; HF then multiplies by its dtype-rounded MuP
buffer. This explicit epilogue retains that intermediate rounding boundary.
"""

import torch
from torch import nn


class FalconH1ProjectionScale(nn.Module):
    def __init__(self, projection, sizes, multipliers):
        super().__init__()
        self.projection = projection
        self.register_buffer(
            "scale", torch.cat([torch.full((size,), value) for size, value in zip(sizes, multipliers)]),
            persistent=False,
        )

    @property
    def weight(self):
        return self.projection.weight

    @property
    def bias(self):
        return self.projection.bias

    def forward(self, hidden):
        return self.projection(hidden) * self.scale
