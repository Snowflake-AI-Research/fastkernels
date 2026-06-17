"""EmbeddingBag lookup primitive for recommender models."""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingBag(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "sum",
        padding_idx: int | None = None,
        include_last_offset: bool = False,
    ):
        super().__init__()
        self.emb = nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            mode=mode,
            padding_idx=padding_idx,
            include_last_offset=include_last_offset,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.emb(input_ids, offsets, per_sample_weights=per_sample_weights)
