"""T5-style RMSNorm with fp32 variance computation.

Matches HuggingFace's T5LayerNorm exactly: upcasts to fp32 for variance
and rsqrt, then casts back before multiplying by the weight.  This is
numerically distinct from the fused _C.rmsnorm kernel used by other
models (which stays in bf16), but required for bit-exact parity with
the HuggingFace / vllm-omni T5 encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
