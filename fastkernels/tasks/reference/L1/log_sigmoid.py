"""Log-sigmoid activation: log(1 / (1 + exp(-x))) = -softplus(-x).

Numerically stable wrapper around F.logsigmoid. Used by GLA's gk gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.logsigmoid(x)
