"""VisionRotaryEmbedding's grid angles with MLCD's learned class-angle prefix."""

from __future__ import annotations

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding


class LearnedPrefixVisionRotaryEmbedding(VisionRotaryEmbedding):
    """Retain grid outer-product/gather/trigonometry, inserting learned angles first.

    The parent caches fixed grid trigonometry. MLCD joins a learned class angle
    to the grid angles before trigonometry, with the inverse frequencies in FP32.
    Build only the current grid on the buffer's device and retain that boundary.
    """

    def __init__(self, rotary_dim: int):
        nn.Module.__init__(self)
        inv_freq = 1 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, height: int, width: int, class_angles: torch.Tensor) -> torch.Tensor:
        device = self.inv_freq.device
        hpos = torch.arange(height, device=device).unsqueeze(1).expand(-1, width)
        wpos = torch.arange(width, device=device).unsqueeze(0).expand(height, -1)
        positions = torch.stack((hpos.flatten(), wpos.flatten()), dim=-1)
        seq = torch.arange(max(height, width), device=device, dtype=self.inv_freq.dtype)
        grid_angles = torch.outer(seq, self.inv_freq)[positions].flatten(1)
        angles = torch.cat((class_angles, grid_angles), dim=0)
        angles = torch.cat((angles, angles), dim=-1)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)
