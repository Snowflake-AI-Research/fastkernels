"""WavLM's two-sigmoid variant of the existing MoE gate epilogue."""

import torch
from torch import nn
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid


@triton.jit
def _gated_bias_kernel(logits, constant, bias, output, length: tl.constexpr,
                       heads: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dtype = logits.dtype.element_ty
    gate_a = tl.sigmoid(tl.load(logits + row * 2).to(tl.float32)).to(dtype).to(tl.float32)
    gate_b = tl.sigmoid(tl.load(logits + row * 2 + 1).to(tl.float32)).to(dtype).to(tl.float32)
    scale = tl.load(constant + (row // length) % heads).to(tl.float32)
    # Preserve each eager HF operation's output rounding before the next one.
    inner = (gate_b * scale).to(dtype).to(tl.float32)
    inner = (inner - 1.0).to(dtype).to(tl.float32)
    gate = (gate_a * inner).to(dtype).to(tl.float32)
    gate = (gate + 2.0).to(dtype).to(tl.float32)
    values = tl.load(bias + row * length + column, column < length, other=0).to(tl.float32)
    tl.store(output + row * length + column, (gate * values).to(dtype), column < length)


class WavLMGatedPositionBias(nn.Module):
    """Adapt L1 moe_shared_gate_add's per-row gate and column-tile scaling.

    The parent epilogue adds a row-gated shared output to a routed output.
    This variant computes two row-local sigmoid gates and scales positional
    bias, keeping the same row/tile launch structure and no new reduction.
    Inputs are projected sums [B,H,T,2], constants [1,H,1,1], and bias
    [B*H,T,T]. The output has the bias shape. CPU execution is a development
    comparison of this patch's arithmetic; CUDA executes the actual kernel.
    """

    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, projected_sums, constant, position_bias):
        batch, heads, length, gates = projected_sums.shape
        if (gates != 2 or position_bias.shape != (batch * heads, length, length)
                or constant.shape != (1, heads, 1, 1)):
            raise ValueError("Unexpected WavLM gate or position-bias shape")
        if len({projected_sums.dtype, constant.dtype, position_bias.dtype}) != 1:
            raise ValueError("WavLM gate inputs must have matching model dtypes")
        if not projected_sums.is_cuda:
            gate_a, gate_b = self.sigmoid(projected_sums).chunk(2, dim=-1)
            gate = gate_a * (gate_b * constant - 1.0) + 2.0
            return gate.reshape(batch * heads, length, 1) * position_bias
        projected_sums = projected_sums.contiguous()
        position_bias = position_bias.contiguous()
        output = torch.empty_like(position_bias)
        block = min(1024, triton.next_power_of_2(length))
        _gated_bias_kernel[(batch * heads * length, triton.cdiv(length, block))](
            projected_sums, constant, position_bias, output, length, heads,
            BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
        return output
