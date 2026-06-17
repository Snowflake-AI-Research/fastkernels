"""Batched matrix multiply (``torch.bmm``) as an L1 primitive.

``torch.bmm`` is a single cuBLAS GEMM kernel; exposing it as a module
keeps L2 code compliant with the "L1-ops only" rule while making the
kernel explicit and profilable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BatchMatMul(nn.Module):
    """Batched matrix multiply ``(B, N, M) @ (B, M, P) -> (B, N, P)``."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)
