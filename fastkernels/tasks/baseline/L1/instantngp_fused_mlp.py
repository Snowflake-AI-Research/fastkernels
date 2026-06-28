"""InstantNGP fully fused MLP wrapper built on tinycudann."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


DEFAULT_DENSITY_NETWORK_CONFIG = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 1,
}

DEFAULT_RGB_NETWORK_CONFIG = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2,
}


class InstantNGPFullyFusedMLP(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        config: dict | None = None,
        *,
        seed: int = 1337,
    ):
        super().__init__()
        import tinycudann as tcnn

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.config = copy.deepcopy(config or DEFAULT_DENSITY_NETWORK_CONFIG)
        self.seed = seed
        self.network = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            network_config=self.config,
            seed=seed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
