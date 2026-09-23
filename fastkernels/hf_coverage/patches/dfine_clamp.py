"""ReLU's pointwise bound operation with two configurable finite bounds."""
from fastkernels.tasks.baseline.L1.relu import ReLU


class DFineClamp(ReLU):
    """Replace max(x, 0) by min(max(x, lower), upper).

    Retains independent elementwise comparisons and linear input/output storage;
    adds no reduction, data dependency, or communication.
    """
    def forward(self, x, lower, upper):
        return x.clamp(min=lower, max=upper)
