"""Graph propagation utilities for LightGCN."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.sparse_mm import SparseMM


def build_normalized_bipartite_adjacency(
    user_indices: torch.Tensor,
    item_indices: torch.Tensor,
    num_users: int,
    num_items: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = user_indices.device

    user_indices = user_indices.to(device=device, dtype=torch.long)
    item_indices = item_indices.to(device=device, dtype=torch.long)
    item_nodes = item_indices + num_users

    row = torch.cat([user_indices, item_nodes], dim=0)
    col = torch.cat([item_nodes, user_indices], dim=0)
    values = torch.ones(row.size(0), device=device)

    size = num_users + num_items
    adjacency = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        values,
        (size, size),
        device=device,
    ).coalesce()

    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min_(1.0)
    norm = degree.pow(-0.5)
    norm_values = adjacency.values() * norm[adjacency.indices()[0]] * norm[adjacency.indices()[1]]
    normalized = torch.sparse_coo_tensor(
        adjacency.indices(),
        norm_values,
        adjacency.size(),
        device=device,
    ).coalesce()
    return normalized.to_sparse_csr()


class LightGCNPropagation(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.sparse_mm = SparseMM()

    def forward(
        self,
        adjacency: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embedding_sum = all_embeddings

        for _ in range(self.num_layers):
            all_embeddings = self.sparse_mm(adjacency, all_embeddings)
            embedding_sum = embedding_sum + all_embeddings

        final_embeddings = embedding_sum / (self.num_layers + 1)
        num_users = user_embeddings.size(0)
        return final_embeddings[:num_users], final_embeddings[num_users:]
