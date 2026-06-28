"""Fused SiLU-mul + per-token-group FP8 quantization with column-major scales.

Single Triton kernel that:
1. Reads the gate/up pair from the first MoE GEMM output [M, 2*N]
2. Computes SiLU(gate) * up -> [M, N]
3. Quantizes each group of 128 elements to FP8 with UE8M0 power-of-two scales
4. Writes FP8 output and column-major scales

Matches vLLM's ``silu_mul_per_token_group_quant_fp8_colmajor`` exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_GROUP_SIZE = 128


@triton.jit
def _silu_mul_per_token_group_quant_fp8_colmajor(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    M,
    N,
    y_s_col_stride: tl.int64,
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset
    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)


class SiluMulQuantFp8(nn.Module):
    """Fused SiLU-mul + per-token-group FP8 quantization (colmajor scales).

    Stateless wrapper around the Triton kernel
    :func:`_silu_mul_per_token_group_quant_fp8_colmajor`.  Mirrors vLLM's
    ``silu_mul_per_token_group_quant_fp8_colmajor`` exactly.
    """

    def forward(
        self,
        input: torch.Tensor,
        output: torch.Tensor | None = None,
        use_ue8m0: bool = True,
        eps: float = 1e-10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused SiLU-mul + per-token-group FP8 quantization.

        Args:
            input: [M, N] where N = 2 * intermediate_size (gate/up concatenated)
            output: Optional pre-allocated [M, N//2] FP8 output buffer
            use_ue8m0: Use power-of-two (UE8M0) scales for DeepGEMM
            eps: Minimum absmax to avoid division by zero

        Returns:
            (output_fp8, output_scales) where output_fp8 is [M, N//2] in
            float8_e4m3fn and output_scales is [M, (N//2)//128] in float32
            (column-major layout)
        """
        assert input.ndim == 2
        M, N = input.size()
        N_2 = N // 2

        assert M % _GROUP_SIZE == 0, f"M={M} must be divisible by {_GROUP_SIZE}"
        assert N_2 % _GROUP_SIZE == 0, f"N//2={N_2} must be divisible by {_GROUP_SIZE}"

        if output is None:
            output = torch.empty(
                (M, N_2), dtype=torch.float8_e4m3fn, device=input.device,
            )

        output_scales = torch.empty(
            (N_2 // _GROUP_SIZE, M), dtype=torch.float32, device=input.device,
        ).transpose(0, 1)

        BLOCK_M = 8
        BLOCK_N = _GROUP_SIZE
        assert M % BLOCK_M == 0
        assert N_2 % BLOCK_N == 0

        fp8_min = _FP8_INFO.min
        fp8_max = _FP8_INFO.max

        grid = (M // BLOCK_M, N_2 // BLOCK_N)

        _silu_mul_per_token_group_quant_fp8_colmajor[grid](
            input, output, output_scales,
            M, N,
            output_scales.stride(-1),
            eps,
            fp8_min, fp8_max,
            use_ue8m0,
            _GROUP_SIZE, BLOCK_M, BLOCK_N,
        )

        return output, output_scales
