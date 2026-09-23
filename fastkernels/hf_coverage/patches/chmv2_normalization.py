"""CHMv2's positive-score variant of the existing L1 L2Norm operation."""

import torch
from torch import nn


class PositiveL1Norm(nn.Module):
    """Keep channel reduction and broadcast division, change the local norm formula.

    Parent: tasks/baseline/L1/l2_norm.py. Positive scores use sum rather than
    sqrt(sum squares), with HF's offset and denominator protection. This does
    not add a reduction axis or change communication between rows.
    """

    def forward(self, positive_scores):
        # For finite ReLU scores, HF's clamp(-amin(scores), 0, 1e-4) is zero.
        # HF materializes the shift in the input dtype before adding it.
        scores = positive_scores + positive_scores.new_tensor(1e-8)
        denominator = scores.sum(dim=1, keepdim=True)
        denominator = torch.nan_to_num(denominator, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-12)
        return scores / denominator


class PositiveFloor(nn.Module):
    """ReLU pointwise clipping with a positive lower bound instead of zero."""

    def forward(self, value):
        return value.clamp_min(1e-12)
