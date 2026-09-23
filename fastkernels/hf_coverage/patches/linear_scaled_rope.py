"""Uniform frequency scaling with the unchanged L1 rotary forward kernel."""

import torch

from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding


class LinearScaledRotaryEmbedding(RotaryEmbedding):
    """Change only fixed-position cache initialization, preserving HF cast order."""

    def __init__(self, head_dim, max_position_embeddings, rope_theta, factor):
        super().__init__(head_dim, max_position_embeddings, rope_theta)
        inverse = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        inverse = inverse / factor
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        self.cos_sin_cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
