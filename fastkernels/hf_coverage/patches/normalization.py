"""Normalization adaptations of existing FastKernels operations."""

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm


class DebertaLayerNorm(nn.Module):
    """LayerNorm with DeBERTa's affine application after the output dtype cast.

    Parent: FastKernels L1 LayerNorm. Its FP32 normalization reduction is
    unchanged; the affine multiply and add execute in the input dtype after
    normalization, matching pinned HF DebertaLayerNorm.forward.
    """

    def __init__(self, size, eps):
        super().__init__()
        self.normalization = LayerNorm(size, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, hidden_states):
        return self.weight * self.normalization(hidden_states) + self.bias
