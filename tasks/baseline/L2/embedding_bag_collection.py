"""Collection of feature-wise embedding bags for recsys models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.embedding_bag import EmbeddingBag


def _flatten_bag_inputs(
    indices: torch.Tensor,
    offsets: torch.Tensor | None,
    per_sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if offsets is not None:
        flat_indices = indices.reshape(-1) if indices.ndim > 1 else indices
        flat_weights = per_sample_weights.reshape(-1) if per_sample_weights is not None else None
        return flat_indices, offsets, flat_weights

    if indices.ndim != 2:
        raise ValueError("indices must be 2D when offsets are not provided")

    batch_size, bag_size = indices.shape
    offsets = torch.arange(
        0,
        batch_size * bag_size,
        bag_size,
        device=indices.device,
        dtype=torch.long,
    )
    flat_indices = indices.reshape(-1)
    flat_weights = per_sample_weights.reshape(-1) if per_sample_weights is not None else None
    return flat_indices, offsets, flat_weights


class EmbeddingBagCollection(nn.Module):
    def __init__(
        self,
        num_embeddings_per_feature: list[int],
        embedding_dim: int,
        mode: str = "sum",
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.num_embeddings_per_feature = list(num_embeddings_per_feature)
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx

        offsets = []
        running = 0
        for size in self.num_embeddings_per_feature:
            offsets.append(running)
            running += size

        self.embedding_bag = EmbeddingBag(
            num_embeddings=running,
            embedding_dim=embedding_dim,
            mode=mode,
            padding_idx=None if padding_idx is None else padding_idx,
        )
        self.register_buffer(
            "table_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=False,
        )
        self._offset_cache: dict[tuple[str, int | None, int, int, int], torch.Tensor] = {}

    def _get_shared_offsets(
        self,
        *,
        device: torch.device,
        num_features: int,
        batch_size: int,
        bag_size: int,
    ) -> torch.Tensor:
        key = (device.type, device.index, num_features, batch_size, bag_size)
        cached = self._offset_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.arange(
                0,
                num_features * batch_size * bag_size,
                bag_size,
                device=device,
                dtype=torch.long,
            )
            self._offset_cache[key] = cached
        return cached

    def _fast_forward_fixed_bags(
        self,
        sparse_indices: list[torch.Tensor],
        per_sample_weights: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        num_features = len(sparse_indices)
        batch_size, bag_size = sparse_indices[0].shape
        stacked_indices = torch.stack(sparse_indices, dim=0)
        shifted_indices = stacked_indices + self.table_offsets.to(
            device=stacked_indices.device,
        ).view(num_features, 1, 1)
        flat_indices = shifted_indices.reshape(-1)
        shared_offsets = self._get_shared_offsets(
            device=stacked_indices.device,
            num_features=num_features,
            batch_size=batch_size,
            bag_size=bag_size,
        )

        flat_weights = None
        if per_sample_weights is not None and any(weight is not None for weight in per_sample_weights):
            if any(weight is None for weight in per_sample_weights):
                raise ValueError("per_sample_weights must be provided for all features")
            flat_weights = torch.stack(per_sample_weights, dim=0).reshape(-1)

        outputs = self.embedding_bag(flat_indices, shared_offsets, per_sample_weights=flat_weights)
        return outputs.reshape(num_features, batch_size, self.embedding_dim).transpose(0, 1).contiguous()

    def forward(
        self,
        sparse_indices: list[torch.Tensor],
        sparse_offsets: list[torch.Tensor] | None = None,
        per_sample_weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(sparse_indices) != len(self.num_embeddings_per_feature):
            raise ValueError("number of sparse feature tensors must match num_embeddings_per_feature")

        if per_sample_weights is None:
            per_sample_weights = [None] * len(sparse_indices)

        if sparse_offsets is None:
            if sparse_indices and all(indices.ndim == 2 for indices in sparse_indices):
                batch_size, bag_size = sparse_indices[0].shape
                if all(indices.shape == (batch_size, bag_size) for indices in sparse_indices):
                    return self._fast_forward_fixed_bags(sparse_indices, per_sample_weights)
            sparse_offsets = [None] * len(sparse_indices)

        flat_indices_per_feature = []
        flat_offsets_per_feature = []
        flat_weights_per_feature = []
        batch_sizes = []
        running = 0

        table_offsets = self.table_offsets.to(device=sparse_indices[0].device)
        for feature_index, (indices, offsets, weights) in enumerate(
            zip(sparse_indices, sparse_offsets, per_sample_weights, strict=True)
        ):
            flat_indices, flat_offsets, flat_weights = _flatten_bag_inputs(indices, offsets, weights)
            flat_indices_per_feature.append(flat_indices + table_offsets[feature_index])
            flat_offsets_per_feature.append(flat_offsets + running)
            if flat_weights is not None:
                flat_weights_per_feature.append(flat_weights)
            running += flat_indices.numel()
            batch_sizes.append(flat_offsets.numel())

        if len(set(batch_sizes)) != 1:
            raise ValueError("all sparse features must produce the same number of bags")

        merged_indices = torch.cat(flat_indices_per_feature, dim=0)
        merged_offsets = torch.cat(flat_offsets_per_feature, dim=0)
        merged_weights = None
        if flat_weights_per_feature:
            if len(flat_weights_per_feature) != len(flat_indices_per_feature):
                raise ValueError("per_sample_weights must be provided for all features")
            merged_weights = torch.cat(flat_weights_per_feature, dim=0)

        outputs = self.embedding_bag(
            merged_indices,
            merged_offsets,
            per_sample_weights=merged_weights,
        )
        batch_size = batch_sizes[0]
        num_features = len(sparse_indices)
        return outputs.reshape(num_features, batch_size, self.embedding_dim).transpose(0, 1).contiguous()
