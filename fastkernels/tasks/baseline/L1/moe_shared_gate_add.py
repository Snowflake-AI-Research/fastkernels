"""Fused MoE epilogue: ``routed + shared * sigmoid(gate)`` in one launch (L1).

A shared-expert MoE layer finishes with three elementwise steps -- sigmoid of
the per-token shared-expert gate, scaling the shared expert's output by it, and
adding that to the routed experts' output. Run separately that is three kernels
and three round trips over the activation; at batch 1 the launches themselves
dominate, and Qwen3-Next has 48 such layers per decode step.

vLLM gets the same collapse for free from Inductor (it shows up as
``triton_poi_fused_mul_silu_slice_0`` in a Qwen3-Next decode trace), including
keeping the intermediate product in fp32 registers rather than rounding it to
bf16 between the multiply and the add -- which this kernel also does.

At small token counts the gate *projection* can come along too. It is a
``[hidden] -> [1]`` dot product per layer, which cuBLAS serves with a gemv whose
cost is entirely launch latency: 5.33 us to read 4 KiB of weights, 48 times per
Qwen3-Next decode step, or 0.256 ms of a 4.03 ms step. Recomputing the dot per
output tile inside this kernel removes that launch. It is only worth it while
the gemv is latency-bound, so callers pass ``hidden``/``gate_weight`` for small
batches and keep the projection for prefill-sized ones.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _shared_gate_add_kernel(
    routed_ptr,
    shared_ptr,
    gate_ptr,
    hidden_ptr,
    gate_w_ptr,
    out_ptr,
    hidden,
    stride_routed,
    stride_shared,
    stride_out,
    stride_gate,
    stride_hidden,
    FUSE_GATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < hidden

    if FUSE_GATE:
        # gate = hidden_row . gate_weight, in fp32. Recomputed per output tile
        # rather than staged through memory: the whole point is to not launch a
        # separate kernel for it.
        h = tl.arange(0, BLOCK_H)
        hm = h < hidden
        row = tl.load(hidden_ptr + pid_t * stride_hidden + h, mask=hm,
                      other=0.0).to(tl.float32)
        w = tl.load(gate_w_ptr + h, mask=hm, other=0.0).to(tl.float32)
        gate = tl.sum(row * w, axis=0)
    else:
        gate = tl.load(gate_ptr + pid_t * stride_gate).to(tl.float32)
    scale = tl.sigmoid(gate)

    shared = tl.load(shared_ptr + pid_t * stride_shared + d, mask=mask).to(tl.float32)
    routed = tl.load(routed_ptr + pid_t * stride_routed + d, mask=mask).to(tl.float32)
    out = routed + shared * scale
    tl.store(
        out_ptr + pid_t * stride_out + d,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def moe_shared_gate_add(
    routed: torch.Tensor,
    shared: torch.Tensor,
    gate: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    gate_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """``routed + shared * sigmoid(gate)`` for a per-token scalar gate.

    Args:
        routed: [T, H] routed-expert output.
        shared: [T, H] shared-expert output, un-gated.
        gate:   [T, 1] or [T] raw (pre-sigmoid) gate, already projected. May be
                a strided column.
        hidden_states, gate_weight: [T, H] and [H] -- supply these *instead of*
                ``gate`` to have the kernel do the projection itself.

    Returns a fresh [T, H] tensor in ``routed``'s dtype.
    """
    n_tokens, hidden = routed.shape
    out = torch.empty_like(routed)
    if n_tokens == 0:
        return out

    fuse_gate = gate is None
    # Triton needs every pointer argument to be a real tensor even on the branch
    # that never dereferences it, so the unused side aliases ``routed``.
    if fuse_gate:
        if hidden_states is None or gate_weight is None:
            raise ValueError("pass gate, or both hidden_states and gate_weight")
        if hidden_states.shape != (n_tokens, hidden):
            raise ValueError(
                f"hidden_states {tuple(hidden_states.shape)} must match routed "
                f"{(n_tokens, hidden)} for the fused gate projection",
            )
        gate_flat = routed
        stride_gate = 0
        stride_hidden = hidden_states.stride(0)
        gate_w = gate_weight.reshape(-1)
    else:
        gate_flat = gate
        stride_gate = gate.stride(0) if gate.dim() else 0
        hidden_states = routed
        stride_hidden = 0
        gate_w = routed

    block_d = min(1024, triton.next_power_of_2(hidden))
    _shared_gate_add_kernel[(n_tokens, triton.cdiv(hidden, block_d))](
        routed,
        shared,
        gate_flat,
        hidden_states,
        gate_w,
        out,
        hidden,
        routed.stride(0),
        shared.stride(0),
        out.stride(0),
        stride_gate,
        stride_hidden,
        FUSE_GATE=fuse_gate,
        BLOCK_D=block_d,
        BLOCK_H=triton.next_power_of_2(hidden),
        num_warps=8 if fuse_gate else 4,
    )
    return out
