"""DP3 diffusion-step embedding MLP.

Mirrors the ``diffusion_step_encoder`` Sequential in
``diffusion_policy_3d.model.diffusion.conditional_unet1d.ConditionalUnet1D``:

    SinusoidalPosEmb(dsed) -> Linear(dsed, 4*dsed) -> Mish -> Linear(4*dsed, dsed)

This file installs the Sequential layout so checkpoint keys at
``model.diffusion_step_encoder.{1,3}.weight/bias`` load by index.
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.dp3_sinusoidal_pos_emb import DP3SinusoidalPosEmb
from ..L1.linear import Linear
from ..L1.mish import Mish


def build_diffusion_step_encoder(dim: int) -> nn.Sequential:
    return nn.Sequential(
        DP3SinusoidalPosEmb(dim),
        Linear(dim, dim * 4),
        Mish(),
        Linear(dim * 4, dim),
    )
