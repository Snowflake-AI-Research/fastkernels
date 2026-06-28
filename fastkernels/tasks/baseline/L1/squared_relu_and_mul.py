"""SquaredReLU-and-Mul activation for BitNet b1.58's gated FFN.

BitNet replaces SiLU with the squared-ReLU activation introduced by
Primer (Shazeer et al., 2021):

    out = relu(gate)^2 * up

The L2 ``BitNetMLP`` calls this op with the merged ``[gate; up]`` tensor
that comes out of the fused ``gate_up_proj`` BitLinearMerged.  Splitting
along the last dim, applying the activation, and the multiply are all
elementwise so a single Triton kernel suffices.

Mirrors the SOTA reference (``vllm_repo/BitNet/gpu/model.py``::``ffn``):
``F.relu(self.w1(...))**2 * self.w3(...)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _squared_relu_and_mul_kernel(
    x_ptr,             # bf16, (M, 2*d)
    out_ptr,           # bf16, (M, d)
    M, d,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK + tl.arange(0, BLOCK)
    d_mask = offs_d < d

    base = pid_m * 2 * d
    gate = tl.load(x_ptr + base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + base + d + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    g = tl.maximum(gate, 0.0)
    out = (g * g) * up
    tl.store(out_ptr + pid_m * d + offs_d, out.to(out_ptr.dtype.element_ty),
             mask=d_mask)


class SquaredReluAndMul(nn.Module):
    """Fused ``relu(gate)^2 * up`` for the BitNet gated FFN.

    Input shape: ``(..., 2*d)`` where the first half is ``gate`` and the
    second half is ``up`` (matching ``MergedColumnParallelLinear`` /
    ``BitLinearMerged`` output ordering).  Output shape: ``(..., d)``.
    """

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        gf = gate.to(torch.float32)
        gf = torch.clamp(gf, min=0.0)
        return ((gf * gf) * up.to(torch.float32)).to(x.dtype)

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        assert x.is_contiguous()
        in_shape = x.shape
        d = in_shape[-1] // 2
        x_2d = x.reshape(-1, 2 * d)
        M = x_2d.shape[0]
        out = torch.empty((M, d), dtype=x.dtype, device=x.device)
        BLOCK = 256 if d >= 256 else triton.next_power_of_2(d)
        grid = (M, triton.cdiv(d, BLOCK))
        _squared_relu_and_mul_kernel[grid](
            x_2d, out, M, d, BLOCK=BLOCK, num_warps=4,
        )
        return out.reshape(*in_shape[:-1], d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling() or not x.is_cuda:
            return self.forward_native(x)
        return self.forward_cuda(x)
