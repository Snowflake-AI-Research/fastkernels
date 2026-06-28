"""DP3 sparse point-cloud encoder.

PointNet-style per-point MLP + max-pool over points + final projection. No
KNN, no T-Net, no STN — DP3 deliberately keeps the encoder minimal (the paper
found heavier point encoders hurt training stability).

Mirrors ``diffusion_policy_3d.model.vision.pointnet_extractor.PointNetEncoderXYZ``.
The ``mlp`` and ``final_projection`` Sequentials are laid out byte-for-byte
the same as the reference so a checkpoint state_dict loads with no key
remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.relu import ReLU


def _mlp_xyz(c0: int, c1: int, c2: int, in_channels: int,
             use_layernorm: bool) -> nn.Sequential:
    return nn.Sequential(
        Linear(in_channels, c0),
        LayerNorm(c0) if use_layernorm else nn.Identity(),
        ReLU(),
        Linear(c0, c1),
        LayerNorm(c1) if use_layernorm else nn.Identity(),
        ReLU(),
        Linear(c1, c2),
        LayerNorm(c2) if use_layernorm else nn.Identity(),
        ReLU(),
    )


def _final_projection(in_dim: int, out_dim: int, final_norm: str,
                      use_projection: bool) -> nn.Module:
    if not use_projection:
        return nn.Identity()
    if final_norm == "layernorm":
        return nn.Sequential(Linear(in_dim, out_dim), LayerNorm(out_dim))
    if final_norm == "none":
        return Linear(in_dim, out_dim)
    raise NotImplementedError(f"final_norm: {final_norm}")


class PointNetEncoderXYZ(nn.Module):
    """3-layer per-point MLP -> max-pool -> projection.

    Args:
        in_channels: must be 3 (XYZ only).
        out_channels: output feature dim (DP3 default 64).
        use_layernorm: whether to apply LayerNorm after each linear.
        final_norm: ``"layernorm"`` or ``"none"`` — apply LayerNorm after final
            projection.
        use_projection: if False, skip the final Linear+LayerNorm and emit
            (B, 256) directly.
    """

    BLOCK_CHANNELS: tuple[int, int, int] = (64, 128, 256)

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        use_projection: bool = True,
    ):
        super().__init__()
        assert in_channels == 3, (
            f"PointNetEncoderXYZ only supports 3 channels, got {in_channels}"
        )
        c0, c1, c2 = self.BLOCK_CHANNELS

        self.mlp = _mlp_xyz(c0, c1, c2, in_channels, use_layernorm)
        self.final_projection = _final_projection(
            c2, out_channels, final_norm, use_projection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_points, 3) float
        Returns:
            (B, out_channels) float
        """
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZRGB(nn.Module):
    """4-layer per-point MLP for XYZRGB inputs (use_pc_color=True path).

    Default DP3 configs do not enable this; provided for completeness.
    """

    BLOCK_CHANNELS: tuple[int, int, int, int] = (64, 128, 256, 512)

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 64,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        use_projection: bool = True,
    ):
        super().__init__()
        c0, c1, c2, c3 = self.BLOCK_CHANNELS

        self.mlp = nn.Sequential(
            Linear(in_channels, c0),
            LayerNorm(c0) if use_layernorm else nn.Identity(),
            ReLU(),
            Linear(c0, c1),
            LayerNorm(c1) if use_layernorm else nn.Identity(),
            ReLU(),
            Linear(c1, c2),
            LayerNorm(c2) if use_layernorm else nn.Identity(),
            ReLU(),
            Linear(c2, c3),
        )
        self.final_projection = _final_projection(
            c3, out_channels, final_norm, use_projection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        x = self.final_projection(x)
        return x
