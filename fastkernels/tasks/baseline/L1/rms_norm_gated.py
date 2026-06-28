"""Gated RMSNorm with element-wise multiplicative gate (L1).

Wraps vLLM's bundled FLA ``rmsnorm_fn`` (``layernorm_guard``) which
implements ``RMSNorm(x) * activation(z)`` in a single fused Triton
kernel. Used by Qwen3-Next's GDN linear-attention output gating where
``norm_before_gate=True`` and ``activation='swish'``.

Keeping this in L1 means the GDN L2 module never imports vLLM directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.fla.ops.layernorm_guard import (
    rmsnorm_fn as _vllm_rmsnorm_fn,
)


class RMSNormGated(nn.Module):
    """Fused gated RMSNorm: ``out = activation(z) * RMSNorm(x, weight)``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 norm_before_gate: bool = True, activation: str = "swish"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return _vllm_rmsnorm_fn(
            x, self.weight, bias=None,
            z=z, eps=self.eps,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )
