"""Jamba's plain (non-MoE) FFN block: fused gate_up_proj + SiluAndMul + down_proj.

Mirrors :class:`L2.llama_mlp.LlamaMLP` exactly -- which is also what
vLLM's ``vllm.model_executor.models.jamba.JambaMLP`` uses (the vLLM
file aliases ``LlamaMLP as JambaMLP``).  Three ops:

    gate_up = MergedColumnParallelLinear(h, [i, i])(x)   # one fused matmul
    activated = SiluAndMul()(gate_up)                    # silu(gate) * up
    out = RowParallelLinear(i, h)(activated)

Layer-uniform with :class:`JambaMoE`: both expose ``forward(x) -> y``
with the same shape so the L3 decoder can switch between them based on
``layers_num_experts[layer_idx]``.

Why this matters for kernel-portability: vLLM's per-step bf16
accumulation order in the FFN block goes through one big
``gate_up_proj`` GEMM followed by a single ``SiluAndMul`` kernel.  An
implementation that runs three separate matmuls (``gate_proj``,
``up_proj``, ``down_proj``) plus a sequence of elementwise
``silu`` + ``mul`` ops produces the *same math* but a different bf16
reduction order, drifting from vLLM's output by ~1e-3 per token --
which compounds to flipped greedy tokens after ~30 decode steps and
shows up as poor match-tokens on the bench.

L1 ops used: ``MergedColumnParallelLinear`` (via the project's
parallel_linear), ``RowParallelLinear``, ``SiluAndMul``.
"""

from __future__ import annotations

import torch.nn as nn

from ..L1.silu_and_mul import SiluAndMul
from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear


class JambaMLP(nn.Module):
    """Jamba dense FFN.  Identical structure to :class:`LlamaMLP`."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)
