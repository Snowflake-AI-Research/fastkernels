"""InstantNGP direction encoding wrapper built on tinycudann."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


DEFAULT_DIRECTION_ENCODING_CONFIG = {
    "otype": "Composite",
    "nested": [
        {
            "n_dims_to_encode": 3,
            "otype": "SphericalHarmonics",
            "degree": 4,
        },
        {
            "otype": "Identity",
        },
    ],
}


class InstantNGPDirectionEncoding(nn.Module):
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
        self.config = copy.deepcopy(config or DEFAULT_DIRECTION_ENCODING_CONFIG)
        self.seed = seed
        self.dtype = dtype
        self.encoding = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config=self.config,
            seed=seed,
            dtype=dtype,
        )
        self.n_output_dims = self.encoding.n_output_dims

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        return self.encoding(directions)
