"""ReLU6's independent clipping operation with ZoeDepth's probability bounds."""

from torch import nn
from torch.nn import functional as F


class ProbabilityClip(nn.Module):
    def forward(self, probability):
        return F.hardtanh(probability, min_val=1e-4, max_val=1.0)
