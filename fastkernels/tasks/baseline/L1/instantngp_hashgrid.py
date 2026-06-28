"""InstantNGP hash-grid encoding wrapper built on tinycudann."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


DEFAULT_HASHGRID_CONFIG = {
    "otype": "HashGrid",
    "n_levels": 8,
    "n_features_per_level": 4,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 2.4380273084089506,
}


class InstantNGPHashGrid(nn.Module):
    def __init__(
        self,
        n_input_dims: int = 3,
        config: dict | None = None,
        *,
        seed: int = 1337,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        import tinycudann as tcnn

        self.n_input_dims = n_input_dims
        self.config = copy.deepcopy(config or DEFAULT_HASHGRID_CONFIG)
        self.seed = seed
        self.dtype = dtype
        self.encoding = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config=self.config,
            seed=seed,
            dtype=dtype,
        )
        self.n_output_dims = self.encoding.n_output_dims

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.encoding(positions)
