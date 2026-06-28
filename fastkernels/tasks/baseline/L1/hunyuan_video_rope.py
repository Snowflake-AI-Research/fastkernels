"""3D rotary position embedding frequency generator for HunyuanVideo-1.5.

Builds a (T', H', W') meshgrid after patch division and computes per-axis
1D rotary embeddings using diffusers' ``get_1d_rotary_pos_embed``.  The
three axes are concatenated along the frequency dimension to produce
(cos, sin) tensors consumed by ``DiffusionRoPE``.

Mirrors vllm-omni's ``HunyuanVideo15RotaryPosEmbed`` in
``vllm_omni/diffusion/models/hunyuan_video/hunyuan_video_15_transformer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.embeddings import get_1d_rotary_pos_embed


class HunyuanVideo15RotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: list[int], theta: float = 256.0) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        grid = torch.stack(grid, dim=0)  # [3, T', H', W']

        freqs = []
        for i in range(3):
            freq_cis = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=False)
            freqs.append((freq_cis.real, freq_cis.imag))

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1).float()
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1).float()
        return freqs_cos, freqs_sin
