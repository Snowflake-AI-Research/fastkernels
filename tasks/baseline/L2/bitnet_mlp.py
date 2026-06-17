"""BitNet MLP: squared-ReLU gated FFN with W1.58A8 BitLinear and sub-norm.

Architecture (per ``microsoft/bitnet-b1.58-2B-4T``)::

    x ─► gate_up_proj (BitLinearMerged) ─► [gate; up]
       ─► SquaredReluAndMul (relu(gate)^2 * up)
       ─► RMSNorm (ffn_sub_norm)
       ─► down_proj (BitLinear)

Mirrors the SOTA reference (``vllm_repo/BitNet/gpu/model.py``) which fuses
``w1`` (gate) and ``w3`` (up) into a single ``w13`` projection.  The
HuggingFace checkpoint stores ``gate_proj`` and ``up_proj`` as separate
tensors; ``packed_modules_mapping`` in ``BitNetForCausalLM`` routes them
into this fused parameter at load time.

Composition: the only L1 ops invoked here are :class:`BitLinear`,
:class:`BitLinearMerged`, :class:`RMSNorm`, and :class:`SquaredReluAndMul`
(no ``torch.nn`` or ``torch.nn.functional`` calls).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.bitnet_linear import BitLinear, BitLinearMerged
from ..L1.bitnet_rms_norm import BitNetRMSNorm as RMSNorm
from ..L1.squared_relu_and_mul import SquaredReluAndMul


class BitNetMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 rms_norm_eps: float = 1e-5):
        super().__init__()
        # Fused gate+up: shard 0 = gate, shard 1 = up (matches the
        # ordering produced by ``MergedColumnParallelLinear`` and the
        # SOTA ``w13`` tensor).
        self.gate_up_proj = BitLinearMerged(
            hidden_size, [intermediate_size, intermediate_size], bias=False,
        )
        self.act_fn = SquaredReluAndMul()
        self.ffn_sub_norm = RMSNorm(intermediate_size, eps=rms_norm_eps)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        inner = self.act_fn(gate_up)
        inner = self.ffn_sub_norm(inner)
        return self.down_proj(inner)
