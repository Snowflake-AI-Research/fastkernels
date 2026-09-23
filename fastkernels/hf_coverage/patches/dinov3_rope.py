"""DINOv3RoPE with HF's inverse-frequency representation and dtype boundary."""

from __future__ import annotations

import math

import torch

from fastkernels.tasks.baseline.L1.dinov3_rope import DINOv3RoPE, _make_coords_dinov3


class HFDINOv3RoPE(DINOv3RoPE):
    """Keep the coordinate/grid/trigonometric computation, using native FP32 inv_freq."""

    def __init__(self, dim: int, temperature: float):
        super().__init__(dim=dim, temperature=temperature)
        del self.periods
        self.register_buffer("inv_freq", 1 / temperature ** torch.arange(
            0, 1, 4 / dim, dtype=torch.float32,
        ), persistent=False)

    def _create_embed(self, feat_shape) -> torch.Tensor:
        coords = _make_coords_dinov3(*feat_shape, device=self.inv_freq.device)
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)
