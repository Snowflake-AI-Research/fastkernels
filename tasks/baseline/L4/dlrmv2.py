"""DLRMv2-style recommendation model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..L2.dlrm_interaction import DLRMFeatureInteraction
from ..L2.embedding_bag_collection import EmbeddingBagCollection
from ..L2.recsys_mlp import RecsysMLP


@dataclass
class DLRMv2Config:
    num_dense_features: int = 13
    num_embeddings_per_feature: list[int] = field(default_factory=lambda: [1024] * 26)
    embedding_dim: int = 64
    bottom_mlp_dims: list[int] = field(default_factory=lambda: [512, 256, 64])
    top_mlp_dims: list[int] = field(default_factory=lambda: [512, 256, 1])
    embedding_bag_mode: str = "sum"


class DLRMv2(nn.Module):
    def __init__(self, config: DLRMv2Config):
        super().__init__()
        self.config = config

        if not config.bottom_mlp_dims or config.bottom_mlp_dims[-1] != config.embedding_dim:
            raise ValueError("bottom_mlp_dims must end with embedding_dim")
        if not config.top_mlp_dims or config.top_mlp_dims[-1] != 1:
            raise ValueError("top_mlp_dims must end with 1")

        self.embedding_bag_collection = EmbeddingBagCollection(
            num_embeddings_per_feature=config.num_embeddings_per_feature,
            embedding_dim=config.embedding_dim,
            mode=config.embedding_bag_mode,
        )
        self.bottom_mlp = RecsysMLP(
            [config.num_dense_features] + config.bottom_mlp_dims,
            activate_last=True,
        )
        feature_count = 1 + len(config.num_embeddings_per_feature)
        num_interactions = feature_count * (feature_count - 1) // 2
        top_input_dim = config.embedding_dim + num_interactions
        self.interaction = DLRMFeatureInteraction()
        self.top_mlp = RecsysMLP([top_input_dim] + config.top_mlp_dims, activate_last=False)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_indices: list[torch.Tensor],
        sparse_offsets: list[torch.Tensor] | None = None,
        per_sample_weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dense_embedding = self.bottom_mlp(dense_features)
        sparse_embeddings = self.embedding_bag_collection(
            sparse_indices,
            sparse_offsets=sparse_offsets,
            per_sample_weights=per_sample_weights,
        )
        interacted = self.interaction(dense_embedding, sparse_embeddings)
        return self.top_mlp(interacted)
