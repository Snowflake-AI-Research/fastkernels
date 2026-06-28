"""Semantic PyTorch reference for fused MoE top-k reduction."""

from __future__ import annotations

import torch
import torch.nn as nn

class MoeSum(nn.Module):
    """Top-k reduction for MoE outputs."""

    def __init__(self):
        super().__init__()
        self._output = None

    def forward(
        self,
        input: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Sum over the topk dimension.

        Args:
            input: [M * topk, D] tensor
            topk: number of experts per token

        Returns:
            output: [M, D] tensor
        """
        total = input.size(0)
        M = total // topk
        D = input.size(1)

        if self._output is None or self._output.size(0) < M or self._output.size(1) < D:
            self._output = torch.empty(M, D, device=input.device, dtype=input.dtype)
        output = self._output[:M, :D]

        output.copy_(input.view(M, topk, D).sum(dim=1))

        return output
