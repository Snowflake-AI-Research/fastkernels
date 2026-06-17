"""Sigmoid top-k routing for Mixture-of-Experts.

Selects top-k experts via torch.topk then applies sigmoid to produce
independent (non-normalized) routing weights. Pre-allocates output buffers
for reuse and CUDA graph compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SigmoidTopK(nn.Module):
    """Top-k selection with sigmoid weights.

    Pre-allocates topk_weights and topk_ids buffers for CUDA graph
    compatibility.
    """

    def __init__(self):
        super().__init__()
        self._topk_weights = None
        self._topk_ids = None

    def _ensure_buffers(self, M, top_k, device):
        if self._topk_weights is None or self._topk_weights.size(0) < M:
            self._topk_weights = torch.empty(
                M, top_k, device=device, dtype=torch.float32,
            )
            self._topk_ids = torch.empty(
                M, top_k, device=device, dtype=torch.int32,
            )

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts with sigmoid weights.

        Args:
            router_logits: [M, num_experts] router scores
            top_k: number of experts per token

        Returns:
            topk_weights: [M, top_k] float32
            topk_ids: [M, top_k] int32
        """
        M = router_logits.size(0)
        self._ensure_buffers(M, top_k, router_logits.device)
        topk_weights = self._topk_weights[:M]
        topk_ids = self._topk_ids[:M]
        scores, indices = torch.topk(router_logits, top_k, dim=-1)
        topk_weights.copy_(torch.sigmoid(scores.float()))
        topk_ids.copy_(indices.to(torch.int32))
        return topk_weights, topk_ids
