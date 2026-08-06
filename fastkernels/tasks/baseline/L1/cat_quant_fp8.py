"""Fused concat + per-tensor FP8 quantization of the absorbed MLA query.

Replaces ``torch.cat([q_absorbed, q_pe], -1)`` followed by
``ops.scaled_fp8_quant`` with a single Triton kernel that reads the two halves
and writes the packed FP8 result directly.

This is what vLLM gets for free. Its ``_DecodeConcatQuantFP8``
(``model_executor/layers/attention/mla_attention.py:1262``) still calls
``torch.cat``, but it is constructed with ``compile_native=True``, so Inductor
fuses cat/reshape/quant/view into one kernel -- the
``triton_poi_fused__to_copy_cat_clamp_mul_reciprocal`` entry in a vLLM decode
profile. fastkernels' MLA path lives inside opaque custom ops that Inductor never
sees, so the fusion has to be written by hand.

Measured on GLM-5.2-NVFP4 decode: the un-fused form cost 28.5 ms of
``CatArrayBatchedCopy`` per 127 steps (vLLM: none at all) plus one extra kernel
launch per layer, and the launch bubbles matter as much as the copy -- at bs=1
fastkernels was issuing 2,818 kernels per step against vLLM's 2,292.

Arithmetic matches ``torch.ops._C.static_scaled_fp8_quant``: multiply by the
*reciprocal* of the scale (vLLM's Inductor kernel name spells this out --
``mul_reciprocal``), clamp to the e4m3 range, then round-to-nearest cast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)


@triton.jit
def _cat_quant_fp8_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    scale_ptr,
    a_stride_n: tl.int64,
    a_stride_h: tl.int64,
    b_stride_n: tl.int64,
    b_stride_h: tl.int64,
    fp8_min,
    fp8_max,
    H: tl.constexpr,
    DA: tl.constexpr,
    DB: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """One program per (token, head) row of the output.

    ``a`` is the absorbed q_nope, which reaches here as a ``bmm(...).transpose``
    view, and ``b`` is a trailing slice of the projected query -- neither is
    contiguous across (token, head), so both take explicit strides. The last
    dimension is contiguous in both cases, so each half is a single vector load.
    """
    row = tl.program_id(0).to(tl.int64)
    n = row // H
    h = row % H

    # Reciprocal once per row, not per element.
    inv_scale = 1.0 / tl.load(scale_ptr)
    out_row = out_ptr + row * (DA + DB)

    offs_a = tl.arange(0, BLOCK_A)
    mask_a = offs_a < DA
    a = tl.load(a_ptr + n * a_stride_n + h * a_stride_h + offs_a,
                mask=mask_a, other=0.0).to(tl.float32)
    a = tl.minimum(tl.maximum(a * inv_scale, fp8_min), fp8_max)
    tl.store(out_row + offs_a, a.to(out_ptr.dtype.element_ty), mask=mask_a)

    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < DB
    b = tl.load(b_ptr + n * b_stride_n + h * b_stride_h + offs_b,
                mask=mask_b, other=0.0).to(tl.float32)
    b = tl.minimum(tl.maximum(b * inv_scale, fp8_min), fp8_max)
    tl.store(out_row + DA + offs_b, b.to(out_ptr.dtype.element_ty), mask=mask_b)


class CatQuantFP8(nn.Module):
    """Concat two ``[N, H, D*]`` halves and per-tensor FP8-quantize in one pass."""

    def forward(
        self,
        a: torch.Tensor,       # [N, H, DA] -- absorbed q_nope
        b: torch.Tensor,       # [N, H, DB] -- q_pe
        scale: torch.Tensor,   # [1] fp32, static per-tensor
    ) -> torch.Tensor:
        assert a.shape[:2] == b.shape[:2], (a.shape, b.shape)
        assert a.stride(-1) == 1 and b.stride(-1) == 1, "last dim must be dense"
        n_tok, n_head, d_a = a.shape
        d_b = b.shape[2]

        out = torch.empty((n_tok, n_head, d_a + d_b),
                          dtype=torch.float8_e4m3fn, device=a.device)
        rows = n_tok * n_head
        if rows == 0:
            return out
        _cat_quant_fp8_kernel[(rows,)](
            a, b, out, scale,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            _FP8_INFO.min, _FP8_INFO.max,
            H=n_head, DA=d_a, DB=d_b,
            BLOCK_A=triton.next_power_of_2(d_a),
            BLOCK_B=triton.next_power_of_2(d_b),
            num_warps=4,
        )
        return out
