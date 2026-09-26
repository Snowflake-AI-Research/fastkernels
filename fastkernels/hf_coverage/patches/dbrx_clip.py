"""Extract the existing SiLU-and-Mul fixed symmetric clipping component.

Parent: tasks/baseline/L1/silu_and_mul.cu, swigluoai_and_mul (line 1060),
whose up branch computes fmaxf(fminf(up, limit), -limit). DBRX needs that
same independent two-sided mapping before QKV splitting, without the
parent gate activation/product. F.hardtanh expresses this mapping for
finite inputs and preserves dtype/shape. There is no new reduction,
communication, dependency or expanded activation storage.
"""
from torch import nn
from torch.nn import functional as F


class DbrxClip(nn.Module):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def forward(self, values):
        return values if self.limit is None else F.hardtanh(values, -self.limit, self.limit)
