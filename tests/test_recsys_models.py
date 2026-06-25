#!/usr/bin/env python3
"""Smoke tests for DLRMv2 and LightGCN."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import torch


def _bootstrap_local_package() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fastkernels",
        os.path.join(root, "__init__.py"),
        submodule_search_locations=[root],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastkernels"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_bootstrap_local_package()

from fastkernels.tasks.baseline.L4.dlrmv2 import DLRMv2, DLRMv2Config
from fastkernels.tasks.baseline.L4.lightgcn import LightGCN, LightGCNConfig


class RecsysSmokeTests(unittest.TestCase):
    def test_dlrmv2_forward(self) -> None:
        config = DLRMv2Config(
            num_dense_features=8,
            num_embeddings_per_feature=[32, 64, 48],
            embedding_dim=16,
            bottom_mlp_dims=[32, 16],
            top_mlp_dims=[32, 1],
        )
        model = DLRMv2(config).eval()

        batch_size = 4
        dense_features = torch.randn(batch_size, config.num_dense_features)
        sparse_indices = [
            torch.randint(0, table_size, (batch_size, 3), dtype=torch.long)
            for table_size in config.num_embeddings_per_feature
        ]

        logits = model(dense_features, sparse_indices)
        self.assertEqual(logits.shape, (batch_size, 1))

    def test_lightgcn_forward(self) -> None:
        config = LightGCNConfig(num_users=6, num_items=10, embedding_dim=8, num_layers=2)
        model = LightGCN(config).eval()

        user_ids = torch.tensor([0, 1, 3, 5], dtype=torch.long)
        item_ids = torch.tensor([1, 4, 7, 8], dtype=torch.long)
        edge_users = torch.tensor([0, 0, 1, 2, 3, 4, 5], dtype=torch.long)
        edge_items = torch.tensor([1, 2, 4, 5, 7, 8, 9], dtype=torch.long)
        adjacency = model.build_adjacency(
            edge_users, edge_items, config.num_users, config.num_items,
        )

        scores = model(user_ids, item_ids, adjacency)
        self.assertEqual(scores.shape, (user_ids.numel(),))

        user_embeddings, item_embeddings = model.get_user_item_embeddings(adjacency)
        self.assertEqual(user_embeddings.shape, (config.num_users, config.embedding_dim))
        self.assertEqual(item_embeddings.shape, (config.num_items, config.embedding_dim))


if __name__ == "__main__":
    unittest.main()
