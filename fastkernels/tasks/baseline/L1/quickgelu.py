"""QuickGELU activation: x * sigmoid(1.702 * x).

Approximation of GELU used in Qwen2-VL vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
