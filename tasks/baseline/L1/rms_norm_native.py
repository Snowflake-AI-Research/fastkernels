"""Pure-PyTorch RMSNorm L1 op (fp32-internal, autograd-friendly).

Distinct from :class:`L1.rms_norm.RMSNorm`:

  - That op dispatches to a CUDA kernel (vLLM's ``_C.rms_norm`` or fastkernels's
    ``fastkernels_norm.rmsnorm``) by default. The kernel is empirically incorrect
    for some hidden sizes (verified wrong output at hidden=16 and hidden=80,
    correct at 32 / 64 / 128) and has no autograd backward registered.
  - This op stays in pure PyTorch, computes variance + rsqrt in fp32 for
    numerical stability, and casts back to the input dtype before the
    weight multiply. Matches JAX/Equinox's ``nn.RMSNorm`` with
    ``promote_dtype(x, weight, dtype=compute_dtype)`` semantics.

Use this op (not :class:`RMSNorm`) wherever:
  - the head_dim or hidden size may not be a multiple of 32 (e.g. TTT-E2E
    qk_norm at head_dim=16, Llama-3 3B-style head_dim=80), OR
  - the norm sits in a ``torch.func.grad`` gradient path (e.g. an inner-loop
    SGD over a subset of params).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNormNative(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xn = xf * torch.rsqrt(var + self.eps)
        return xn.to(orig_dtype) * self.weight
