"""Whisper MLP block: fc1 -> activation -> fc2.

Uses GELU activation (not SwiGLU like Llama). Matches vLLM's WhisperMLP.
"""

from __future__ import annotations

import torch.nn as nn

from .parallel_linear import ColumnParallelLinear, RowParallelLinear
from ..L1.gelu import GELU


class WhisperMLP(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(embed_dim, ffn_dim, bias=True)
        self.fc2 = RowParallelLinear(ffn_dim, embed_dim, bias=True)
        self.act_fn = GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        return self.fc2(x)
