"""SinusoidalEmbed with VoxtralRealtime's native frequency and dtype contract."""

import math
import torch
from fastkernels.tasks.baseline.L1.sinusoidal_embed import SinusoidalEmbed


class VoxtralRealtimeTimeEmbedding(SinusoidalEmbed):
    """Keep independent frequency products and trigonometric output storage.

    The parent promotes products/trigonometry to FP32 and emits sine first.
    This adaptation retains the input dtype and emits cosine first, using the
    checkpoint's exp(-log(theta)*index/half_width) frequency schedule.
    """

    def __init__(self, width):
        super().__init__(width)
        self.sinusoid_freq = torch.exp(-math.log(10000.0) * torch.arange(width // 2).float() / (width // 2))

    def forward(self, timestep):
        angles = timestep[:, None] * self.sinusoid_freq.to(timestep.dtype)[None]
        return torch.cat((angles.cos(), angles.sin()), dim=-1)
