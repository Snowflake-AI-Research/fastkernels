"""RTDetrV2 inverse_sigmoid's pointwise division→log, with explicit operands.

Parent: tasks/baseline/L3/rtdetrv2_decoder.py::inverse_sigmoid.
The parent constructs clamped x and (1-x) operands. This semantic adaptation
accepts numerator/denominator directly and removes unit-interval clamps:
positive ratios map to finite logs; zero maps to -inf, negatives to NaN.
It preserves the actual parent division followed by log and introduces no
reduction, communication, or additional input-dependent storage expansion.
"""
import torch
from torch import nn


class RatioLog(nn.Module):
    def forward(self, numerator, denominator):
        return torch.log(numerator / denominator)
