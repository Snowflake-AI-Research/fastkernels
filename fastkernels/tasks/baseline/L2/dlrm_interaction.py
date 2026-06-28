"""Feature interaction block for DLRM-style recommenders."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import BMM


class DLRMFeatureInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.bmm = BMM()
        self._triu_cache: dict[tuple[str, int | None, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _interaction_indices(self, feature_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, feature_count)
        cached = self._triu_cache.get(key)
        if cached is None or cached[0].device != device:
            cached = torch.triu_indices(
                feature_count,
                feature_count,
                offset=1,
                device=device,
            )
            cached = (cached[0], cached[1])
            self._triu_cache[key] = cached
        return cached

    def forward(
        self,
        dense_embedding: torch.Tensor,
        sparse_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if dense_embedding.ndim != 2:
            raise ValueError("dense_embedding must have shape (batch, dim)")
        if sparse_embeddings.ndim != 3:
            raise ValueError("sparse_embeddings must have shape (batch, features, dim)")

        features = torch.cat([dense_embedding.unsqueeze(1), sparse_embeddings], dim=1)
        interaction = self.bmm(features, features.transpose(1, 2))
        feature_count = interaction.size(1)
        row_idx, col_idx = self._interaction_indices(feature_count, interaction.device)
        pairwise = interaction[:, row_idx, col_idx]
        return torch.cat([dense_embedding, pairwise], dim=1)
