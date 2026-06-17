"""OAI SwiGLU activation: (up + 1) * gate * sigmoid(alpha * gate) with clamping.

Used by GPT-OSS MoE experts. Differs from standard SwiGLU (SiLU-and-Mul) in
that it uses a shifted up-projection (up + 1), a tuned alpha constant, and
clamps gate/up values for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SWIGLU_ALPHA = 1.702


class SwigluOai(nn.Module):
    """Clamped OAI SwiGLU: (up + 1) * gate * sigmoid(alpha * gate)."""

    def __init__(self, swiglu_limit: float = 7.0):
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * _SWIGLU_ALPHA)
        return (up + 1) * glu
