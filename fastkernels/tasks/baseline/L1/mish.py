"""Mish activation: x * tanh(softplus(x)).

Used by DP3's 1-D conditional U-Net (Conv1dBlock, FiLM cond_encoder, and the
diffusion-step-embedding MLP).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x)
