"""Fused indexer weight scaling: one kernel for the ``wp_out * q_scale * ...`` chain.

`SparseAttnIndexer` computes

    weights = (wp_out.unsqueeze(-1) * q_scale * softmax_scale * n_head ** -0.5
               ).squeeze(-1)

which is four separate elementwise launches per indexer compute layer -- one
tensor multiply and three scalar multiplies. At 21 compute layers that is ~84
launches per decode step, measured at ~89 us/step of GPU time, and vLLM has none
of them: it writes the same expression but its model code runs under Inductor,
which collapses the chain into a single kernel.

The multiplies are applied here in the **same order and the same precision** as
the PyTorch chain -- `bf16 wp_out` promoted to fp32 by the fp32 `q_scale`, then
the two scalars folded in one at a time. Reassociating them (pre-multiplying
``softmax_scale * n_head ** -0.5`` into one constant) would be one fewer multiply
but is *not* bit-neutral, and the indexer output drives top-k selection where a
single flipped index cascades, so the order is preserved exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _indexer_weights_kernel(
    wp_ptr,
    qs_ptr,
    out_ptr,
    wp_stride_m: tl.int64,
    qs_stride_m: tl.int64,
    out_stride_m: tl.int64,
    softmax_scale,
    head_scale,
    N_HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_HEAD

    # `wp_out` is a column slice of the fused wk/wp GEMM output, so its row
    # stride is the full GEMM width rather than N_HEAD.
    wp = tl.load(wp_ptr + m * wp_stride_m + offs, mask=mask, other=0.0)
    qs = tl.load(qs_ptr + m * qs_stride_m + offs, mask=mask, other=0.0)

    # Order matches the PyTorch chain: promote to fp32 via the fp32 q_scale, then
    # apply the scalars separately.
    w = wp.to(tl.float32) * qs
    w = w * softmax_scale
    w = w * head_scale
    tl.store(out_ptr + m * out_stride_m + offs, w, mask=mask)


class IndexerWeights(nn.Module):
    """``(wp_out * q_scale * softmax_scale * n_head**-0.5)`` in one kernel."""

    def forward(
        self,
        wp_out: torch.Tensor,     # [M, n_head], bf16 slice of the fused GEMM
        q_scale: torch.Tensor,    # [M, n_head, 1] fp32
        softmax_scale: float,
        head_scale: float,
    ) -> torch.Tensor:
        n_tok, n_head = wp_out.shape
        assert q_scale.shape[:2] == (n_tok, n_head), q_scale.shape
        assert wp_out.stride(1) == 1 and q_scale.stride(1) == 1, "heads must be dense"
        out = torch.empty((n_tok, n_head), dtype=torch.float32,
                          device=wp_out.device)
        if n_tok == 0:
            return out
        _indexer_weights_kernel[(n_tok,)](
            wp_out, q_scale, out,
            wp_out.stride(0), q_scale.stride(0), out.stride(0),
            softmax_scale, head_scale,
            N_HEAD=n_head, BLOCK=triton.next_power_of_2(n_head),
            num_warps=1,
        )
        return out
