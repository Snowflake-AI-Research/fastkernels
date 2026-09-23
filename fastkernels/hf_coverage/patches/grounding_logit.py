"""Clipping-boundary variant of RT-DETR's existing inverse_sigmoid callable."""

import torch
from torch import nn


class GroundingLogit(nn.Module):
    """Clamp the input to [eps, 1-eps] before evaluating its log odds.

    Parent: L3/rtdetrv2_decoder.py::inverse_sigmoid. That parent separately
    clamps the numerator and denominator; HF Grounding DINO instead clips the
    input. The difference is observable on actual model values at one. This
    keeps the pointwise dependencies and storage, with PyTorch's actual logit
    implementation timed, and does not claim an unimplemented CUDA speedup.
    """
    def forward(self, values):
        return torch.special.logit(values, eps=1e-5)
