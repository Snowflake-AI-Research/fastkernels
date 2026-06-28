"""LightGCN recommendation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L2.lightgcn_propagation import LightGCNPropagation, build_normalized_bipartite_adjacency


@dataclass
class LightGCNConfig:
    num_users: int
    num_items: int
    embedding_dim: int = 64
    num_layers: int = 3


class LightGCN(nn.Module):
    def __init__(self, config: LightGCNConfig):
        super().__init__()
        self.config = config
        self.user_embedding = Embedding(config.num_users, config.embedding_dim)
        self.item_embedding = Embedding(config.num_items, config.embedding_dim)
        self.propagation = LightGCNPropagation(config.num_layers)

    def get_initial_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_embedding.emb.weight, self.item_embedding.emb.weight

    def get_user_item_embeddings(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_embeddings, item_embeddings = self.get_initial_embeddings()
        return self.propagation(adjacency, user_embeddings, item_embeddings)

    def score(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings, item_embeddings = self.get_user_item_embeddings(adjacency)
        return (user_embeddings[user_ids] * item_embeddings[item_ids]).sum(dim=-1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        return self.score(user_ids, item_ids, adjacency)

    @staticmethod
    def build_adjacency(
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        num_users: int,
        num_items: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return build_normalized_bipartite_adjacency(
            user_indices, item_indices, num_users, num_items, device=device,
        )
