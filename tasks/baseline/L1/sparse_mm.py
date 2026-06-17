"""Sparse matrix multiply primitive for graph recommenders."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseMM(nn.Module):
    def forward(self, sparse_matrix: torch.Tensor, dense_matrix: torch.Tensor) -> torch.Tensor:
        if sparse_matrix.layout in {
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            return torch.sparse.mm(sparse_matrix, dense_matrix)
        return sparse_matrix @ dense_matrix
