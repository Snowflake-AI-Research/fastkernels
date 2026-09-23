"""Dynamic positional tables using FastKernels' existing native rotation."""

import torch

from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding


class InternVLDynamicRotaryEmbedding(RotaryEmbedding):
    """Adapt table preparation only; activation rotation remains the parent op.

    Growth and reset follow HF's documented dynamic-frequency behavior. The
    pinned reference has a reset-bookkeeping defect; cross-context reuse after
    a long input is recorded separately from single-context compatibility.
    """

    def __init__(self, head_dim, max_position_embeddings, rope_theta, factor):
        super().__init__(head_dim, max_position_embeddings, rope_theta)
        self.original_limit = max_position_embeddings
        self.cached_limit = max_position_embeddings
        self.theta, self.factor = rope_theta, factor
        # HF loads its nonpersistent inverse frequencies on CPU, then forms
        # angles on the execution device. GPU pow differs by one FP32 ULP for
        # this checkpoint; at long positions that crosses a BF16 cosine bin.
        self.cos_sin_cache = self.initial_cache(max_position_embeddings, self.cos_sin_cache.device)

    def initial_cache(self, length, device):
        indices = torch.arange(0, self.head_dim, 2, device="cpu", dtype=torch.float32)
        inverse = (1.0 / (self.theta ** (indices / self.head_dim))).to(device)
        steps = torch.arange(length, device=device, dtype=torch.float32)
        angles = steps[:, None] * inverse[None]
        return torch.cat((angles.cos(), angles.sin()), dim=-1)

    def prepare_positions(self, positions):
        length = positions.max() + 1
        if length < self.original_limit and self.cached_limit > self.original_limit:
            self.cos_sin_cache = self.initial_cache(self.original_limit, positions.device).to(self.cos_sin_cache.dtype)
            self.cached_limit = self.original_limit
        elif length > self.cached_limit:
            length = torch.maximum(length, length.new_tensor(self.original_limit))
            base = self.theta * ((self.factor * length / self.original_limit) - (self.factor - 1)) ** (
                self.head_dim / (self.head_dim - 2)
            )
            indices = torch.arange(0, self.head_dim, 2, device=positions.device, dtype=torch.float32)
            inverse = 1.0 / (base ** (indices / self.head_dim))
            steps = torch.arange(length, device=positions.device, dtype=torch.float32)
            angles = steps[:, None] * inverse[None]
            self.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(self.cos_sin_cache.dtype)
            self.cached_limit = int(length)

    def forward(self, positions, query, key):
        self.prepare_positions(positions)
        return self.forward_native(positions, query, key, self.head_dim,
                                   self.cos_sin_cache.to(query.dtype))
