"""Pix2Struct's finite-score floor before the existing softmax reduction."""

import torch
from fastkernels.tasks.baseline.L1.softmax import Softmax


class FiniteFloorSoftmax(Softmax):
    """Keep the input-dtype floor, FP32 reduction and input-dtype output."""

    def forward(self, scores):
        floor = torch.finfo(scores.dtype).min
        return super().forward(scores.clamp_min(floor).float()).to(scores.dtype)
