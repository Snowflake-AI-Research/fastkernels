"""AvgPool2d with padded cells excluded from the averaging divisor."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fastkernels.tasks.baseline.L1.avg_pool2d import AvgPool2d


class ExcludePaddingAvgPool2d(AvgPool2d):
    """Preserve window geometry and reduction; divide by valid cell count."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding,
            ceil_mode=self.ceil_mode, count_include_pad=False,
        )
