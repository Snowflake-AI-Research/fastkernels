"""Precision adaptation of the existing RotaryEmbedding operation for ERNIE 4.5."""

import torch

from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding


class FP32RotaryEmbedding(RotaryEmbedding):
    """Keep FP32 angles in the parent's existing mixed-dtype CUDA rotation.

    The parent CUDA rotation, dependencies, and cache strategy are
    unchanged. That kernel promotes Q/K values to FP32 internally and casts
    only its final stores. Its cache dtype is independent of Q/K; preserve it
    instead of applying the ordinary parent's cache-to-Q-dtype conversion.
    Construct and attach this operation after moving the rest of the model so
    its FP32 cache is never rounded by a blanket model dtype conversion.
    """

    def forward_cuda(self, positions, query, key):
        torch.ops.fastkernels_rope.rotary_embedding(
            positions, query, key, self.head_dim, self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def forward(self, positions, query, key):
        return self.forward_cuda(positions, query, key)
