"""DINOv3 2D rotary position embedding (L1).

Temperature-based 2D RoPE with rotate-half layout matching the DINOv3
position encoding scheme. Produces concatenated [sin, cos] embeddings
for each spatial position.

Reference: timm/layers/pos_embed_sincos.py RotaryEmbeddingDinoV3
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _make_coords_dinov3(
    height: int,
    width: int,
    normalize_coords: str = "separate",
    grid_offset: float = 0.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build 0.5-centered normalized coordinate grid in [-1, 1].

    Returns (H*W, 2) tensor.
    """
    coords_h = torch.arange(0.5, height, device=device, dtype=torch.float32) + grid_offset
    coords_w = torch.arange(0.5, width, device=device, dtype=torch.float32) + grid_offset

    if normalize_coords == "separate":
        h_denom = float(height)
        w_denom = float(width)
    elif normalize_coords == "max":
        h_denom = w_denom = float(max(height, width))
    elif normalize_coords == "min":
        h_denom = w_denom = float(min(height, width))
    else:
        raise ValueError(f"Unknown normalize_coords: {normalize_coords}")

    coords_h = (coords_h / h_denom).to(dtype)
    coords_w = (coords_w / w_denom).to(dtype)

    coords = torch.stack(
        torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
    )  # (H, W, 2)
    coords = coords.flatten(0, 1)  # (H*W, 2)
    coords = 2.0 * coords - 1.0  # map to [-1, 1]
    return coords


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate-half: split at midpoint, negate first half, concatenate reversed."""
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rot_embed_cat(
    x: torch.Tensor,
    emb: torch.Tensor,
    half: bool = True,
) -> torch.Tensor:
    """Apply concatenated [sin, cos] rotary embedding to x.

    Args:
        x: (..., D) tensor.
        emb: (..., 2*D) tensor with [sin, cos] concatenated.
        half: If True, use rotate-half layout; otherwise interleaved.
    """
    sin_emb, cos_emb = emb.chunk(2, -1)
    if half:
        return x * cos_emb + _rope_rotate_half(x) * sin_emb
    else:
        # Interleaved rotation
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos_emb + rotated * sin_emb


class DINOv3RoPE(nn.Module):
    """DINOv3 2D rotary position embedding.

    Args:
        dim: Head dimension (total RoPE output dim = dim).
        temperature: Base temperature for frequency computation.
        feat_shape: Optional (H, W) to pre-cache embeddings.
        normalize_coords: Coordinate normalization mode.
        grid_offset: Offset for grid coordinates.
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 100.0,
        feat_shape: Optional[List[int]] = None,
        normalize_coords: str = "separate",
        grid_offset: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.normalize_coords = normalize_coords
        self.grid_offset = grid_offset
        self.rotate_half = True
        self.feat_shape = feat_shape

        periods = self._compute_periods()
        self.register_buffer("periods", periods, persistent=False)

        if feat_shape is not None:
            num_pos = feat_shape[0] * feat_shape[1]
            emb = self._create_embed(feat_shape)
            self.register_buffer("pos_embed_cached", emb, persistent=False)
        else:
            self.pos_embed_cached = None

    def _compute_periods(self) -> torch.Tensor:
        dim = self.dim // 4
        exponents = 2.0 * torch.arange(dim, dtype=torch.float32) / (self.dim // 2)
        periods = self.temperature ** exponents
        return periods

    def _create_embed(self, feat_shape: List[int]) -> torch.Tensor:
        H, W = feat_shape
        coords = _make_coords_dinov3(
            H, W,
            normalize_coords=self.normalize_coords,
            grid_offset=self.grid_offset,
        )
        dim = self.dim // 4
        coords = coords[:, :, None].to(dtype=self.periods.dtype, device=self.periods.device)
        angles = 2 * math.pi * coords / self.periods[None, None, :]
        angles = angles.flatten(1)  # (H*W, dim//2)
        angles = angles.tile(2)  # rotate-half layout: (H*W, dim)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat([sin, cos], dim=-1)  # (H*W, 2*dim)

    def get_embed(self, shape: Optional[List[int]] = None) -> torch.Tensor:
        if shape is not None:
            return self._create_embed(shape)
        assert self.pos_embed_cached is not None
        return self.pos_embed_cached

    def update_feat_shape(self, feat_shape: List[int]):
        if self.feat_shape is None or feat_shape != self.feat_shape:
            self.feat_shape = feat_shape
            emb = self._create_embed(feat_shape)
            self.register_buffer("pos_embed_cached", emb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.get_embed(list(x.shape[2:]))
        return apply_rot_embed_cat(x, pos_embed, half=True)
