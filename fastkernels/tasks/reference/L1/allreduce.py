"""Self-contained NCCL allreduce reference.

The production baseline may use a custom IPC allreduce fast path. This reference
keeps only the correctness path: ``torch.distributed.all_reduce``. It is slower
but avoids importing baseline code or compiling baseline-local CUDA sources.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup


_CUSTOM_AR: Optional["CustomAllreduce"] = None


def set_custom_ar(ar):
    global _CUSTOM_AR
    _CUSTOM_AR = ar


def get_custom_ar():
    return _CUSTOM_AR


class AllReduce(nn.Module):
    def forward(self, tensor):
        dist.all_reduce(tensor)
        return tensor


class CustomAllreduce:
    """Compatibility shim for callers expecting the baseline custom AR API."""

    disabled = True

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
    ) -> None:
        del group, device, max_size

    def capture(self):
        return nullcontext()

    def custom_all_reduce(self, input: torch.Tensor) -> None:
        del input
        return None

    def close(self) -> None:
        pass

__all__ = ["AllReduce", "CustomAllreduce", "get_custom_ar", "set_custom_ar"]
