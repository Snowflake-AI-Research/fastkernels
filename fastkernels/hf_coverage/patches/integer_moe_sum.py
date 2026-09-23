"""MoeSum's existing generic reduction, selected for signed integer inputs."""

import torch

from fastkernels.tasks.baseline.L1.moe_sum import MoeSum


class IntegerMoeSum(MoeSum):
    """Keep the parent's reduction axis, layout, and reusable output storage.

    MoeSum already calls ATen sum_out for generic group sizes, but its special
    paths for groups 2, 3, 4, and 8 accept only floating types. This adaptation
    selects that same generic backend for int64 at every group size. It keeps
    integer accumulation rather than converting BLT's hash terms to floats.
    """

    def forward(self, values: torch.Tensor, topk: int) -> torch.Tensor:
        if values.dtype != torch.int64:
            raise TypeError("IntegerMoeSum requires torch.int64 inputs")

        rows = values.size(0) // topk
        width = values.size(1)
        if (self._output is None or self._output.size(0) < rows
                or self._output.size(1) < width):
            self._output = values.new_empty(rows, width)
        output = self._output[:rows, :width]
        return torch.sum(values.view(rows, topk, width), dim=1, out=output)
