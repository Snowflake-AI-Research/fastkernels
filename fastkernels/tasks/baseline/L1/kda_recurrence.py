"""Fused recurrent KDA decode kernel.

Thin ``nn.Module`` wrapper around a custom Triton decode kernel used by
Kimi-Linear. The wrapper mirrors existing L1 recurrent-kernel files while the
functional entry point remains available for focused tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.model_executor.layers.fla.ops.op import exp
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["N", "T"])
def _fused_recurrent_kda_chunk_output_decode_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h,
    cu_seqlens,
    state_indices,
    scale,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    if eos <= bos:
        return

    state_idx = tl.load(state_indices + i_n * stride_indices_seq).to(tl.int64)
    if state_idx < 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + (bos * HV + i_hv) * K + o_k
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v

    p_h = h + state_idx * stride_state_token + i_hv * V * K
    h_vk = tl.load(
        p_h + o_v[:, None] * K + o_k[None, :],
        mask=mask_h,
        other=0,
    ).to(tl.float32)

    q_vec = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    k_vec = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    v_vec = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    g_vec = exp(tl.load(p_g, mask=mask_k, other=0).to(tl.float32))

    q_vec = q_vec / tl.sqrt(tl.sum(q_vec * q_vec) + 1e-6)
    k_vec = k_vec / tl.sqrt(tl.sum(k_vec * k_vec) + 1e-6)
    q_vec = q_vec * scale

    h_gated = h_vk * g_vec[None, :]
    v_delta = v_vec - tl.sum(h_gated * k_vec[None, :], axis=1)
    if IS_BETA_HEADWISE:
        beta_vec = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
    else:
        beta_vec = tl.load(p_beta).to(tl.float32)
    v_delta = v_delta * beta_vec

    # chunk_kda reads the initial state contribution as H^T @ (q * exp(g)).
    # The recurrent update above keeps vLLM's canonical [V, K] state layout.
    h_kv = tl.load(
        p_h + o_k[:, None] * K + o_v[None, :],
        mask=mask_k[:, None] & mask_v[None, :],
        other=0,
    ).to(tl.float32)
    initial_out = tl.sum(h_kv * (q_vec * g_vec)[:, None], axis=0)
    h_new = h_gated + v_delta[:, None] * k_vec[None, :]
    update_out = tl.sum(
        (v_delta[:, None] * k_vec[None, :]) * q_vec[None, :],
        axis=1,
    )
    out = initial_out + update_out
    tl.store(p_o, out.to(p_o.dtype.element_ty), mask=mask_v)
    tl.store(
        p_h + o_v[:, None] * K + o_k[None, :],
        h_new.to(h.dtype.element_ty),
        mask=mask_h,
    )


def fused_recurrent_kda_chunk_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode KDA with chunk_kda-compatible output and in-place state update.

    vLLM's fused recurrent KDA updates state in the same layout as chunk_kda,
    but its decode output reads the initial-state term as H @ q. Kimi's
    chunk_kda path reads that term as H^T @ q. This decode-only kernel keeps
    the fused recurrent state update while producing the chunk_kda output
    directly, avoiding a separate state clone and two matmuls per KDA layer.
    """
    if q.shape[0] != 1 or cu_seqlens is None or state_indices is None:
        raise ValueError("fused_recurrent_kda_chunk_output expects varlen decode")
    if q.shape[1] != state_indices.numel():
        raise ValueError("decode kernel expects one token per sequence")
    if k.shape[-1] != v.shape[-1]:
        raise ValueError("Kimi KDA decode expects key/value head dims to match")
    if scale is None:
        scale = k.shape[-1] ** -0.5

    _, total_tokens, num_heads, head_dim = q.shape
    value_dim = v.shape[-1]
    value_heads = v.shape[2]
    block_k = triton.next_power_of_2(head_dim)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    num_v_blocks = triton.cdiv(value_dim, block_v)
    nseq = cu_seqlens.numel() - 1
    stride_indices_seq = state_indices.stride(0)

    out = torch.empty_like(v)
    grid = (num_v_blocks, nseq * value_heads)
    _fused_recurrent_kda_chunk_output_decode_kernel[grid](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        out,
        initial_state,
        cu_seqlens,
        state_indices,
        scale,
        nseq,
        total_tokens,
        num_heads,
        value_heads,
        head_dim,
        value_dim,
        block_k,
        block_v,
        initial_state.stride(0),
        stride_indices_seq,
        beta.ndim == v.ndim,
        num_warps=1,
        num_stages=3,
    )
    return out, initial_state


class FusedRecurrentKDAChunkOutput(nn.Module):
    """Decode-only KDA recurrent kernel with chunk_kda-compatible output."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_recurrent_kda_chunk_output(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            scale=scale,
        )
