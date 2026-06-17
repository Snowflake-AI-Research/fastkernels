"""SwiGLU activation and AdaLN for AlphaFold3 (L2 composites).

SwiGLU: SiLU(linear_a(x)) * linear_b(x)
AdaLN: Adaptive Layer Normalization

Reference: openfold3/core/model/primitives/activations.py SwiGLU
           openfold3/core/model/primitives/normalization.py AdaLN
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.linear_a(x)) * self.linear_b(x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))
