"""Kernel-level InstantNGP field built from tinycudann primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..L1.instantngp_direction_encoding import (
    DEFAULT_DIRECTION_ENCODING_CONFIG,
    InstantNGPDirectionEncoding,
)
from ..L1.instantngp_fused_mlp import (
    DEFAULT_DENSITY_NETWORK_CONFIG,
    DEFAULT_RGB_NETWORK_CONFIG,
    InstantNGPFullyFusedMLP,
)
from ..L1.instantngp_hashgrid import (
    DEFAULT_HASHGRID_CONFIG,
    InstantNGPHashGrid,
)
from ..L1.tensor_ops import Cat, Exp


@dataclass(frozen=True)
class InstantNGPFieldOutput:
    sigma: torch.Tensor
    geo_feat: torch.Tensor
    rgb: torch.Tensor


class InstantNGPField(nn.Module):
    def __init__(
        self,
        *,
        hashgrid_config: dict | None = None,
        direction_encoding_config: dict | None = None,
        density_network_config: dict | None = None,
        rgb_network_config: dict | None = None,
        geo_feat_dims: int = 15,
        seed: int = 1337,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hashgrid_config = hashgrid_config or DEFAULT_HASHGRID_CONFIG
        self.direction_encoding_config = (
            direction_encoding_config or DEFAULT_DIRECTION_ENCODING_CONFIG
        )
        self.density_network_config = density_network_config or DEFAULT_DENSITY_NETWORK_CONFIG
        self.rgb_network_config = rgb_network_config or DEFAULT_RGB_NETWORK_CONFIG
        self.geo_feat_dims = geo_feat_dims
        self.seed = seed
        self.dtype = dtype

        self.position_encoding = InstantNGPHashGrid(
            config=self.hashgrid_config,
            seed=seed,
            dtype=dtype,
        )
        self.density_mlp = InstantNGPFullyFusedMLP(
            n_input_dims=self.position_encoding.n_output_dims,
            n_output_dims=1 + geo_feat_dims,
            config=self.density_network_config,
            seed=seed,
        )
        self.direction_encoding = InstantNGPDirectionEncoding(
            config=self.direction_encoding_config,
            seed=seed,
            dtype=dtype,
        )
        self.rgb_mlp = InstantNGPFullyFusedMLP(
            n_input_dims=self.direction_encoding.n_output_dims + 1 + geo_feat_dims,
            n_output_dims=3,
            config=self.rgb_network_config,
            seed=seed,
        )
        self.cat_features = Cat(dim=-1)
        self.exp = Exp()

    @property
    def n_flat_params(self) -> int:
        return sum(param.numel() for param in self.parameters())

    @torch.no_grad()
    def load_flat_params(self, flat_params: torch.Tensor) -> None:
        if flat_params.ndim != 1:
            raise ValueError(f"Expected 1D flat params, got shape {tuple(flat_params.shape)}")
        if flat_params.numel() != self.n_flat_params:
            raise ValueError(
                f"Flat param size mismatch: expected {self.n_flat_params}, got {flat_params.numel()}"
            )
        offset = 0
        for param in (
            self.density_mlp.network.params,
            self.rgb_mlp.network.params,
            self.position_encoding.encoding.params,
            self.direction_encoding.encoding.params,
        ):
            n = param.numel()
            if n:
                param.copy_(flat_params[offset : offset + n].to(param.device, dtype=param.dtype))
            offset += n

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> InstantNGPFieldOutput:
        density_features = self.density_mlp(self.position_encoding(positions))
        sigma = density_features[..., :1]
        geo_feat = density_features[..., 1:]
        dir_features = self.direction_encoding(directions)
        # instant-ngp feeds the full density MLP output (raw sigma + geo features)
        # followed by the direction encoding into the color MLP.
        rgb_input = self.cat_features([density_features, dir_features])
        rgb = self.rgb_mlp(rgb_input)
        return InstantNGPFieldOutput(sigma=sigma, geo_feat=geo_feat, rgb=rgb)

    def query_density(self, positions: torch.Tensor) -> torch.Tensor:
        density_features = self.density_mlp(self.position_encoding(positions))
        return self.exp(density_features[..., :1].to(torch.float32))
