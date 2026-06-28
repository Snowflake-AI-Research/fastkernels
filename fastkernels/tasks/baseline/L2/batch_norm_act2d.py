"""BatchNorm2d followed by an optional activation."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..L1.batch_norm2d import BatchNorm2d


class BatchNormAct2d(BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        act_layer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.act = act_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.act is not None:
            x = self.act(x)
        return x
