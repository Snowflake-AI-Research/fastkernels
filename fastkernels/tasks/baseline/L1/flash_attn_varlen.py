"""Variable-length Flash Attention (no KV cache lookup).

Thin ``nn.Module`` wrapper around ``flash_attn_varlen_func`` from vLLM's
bundled FlashAttention build, at the version vLLM would select for this
device (see :mod:`fa_utils`).  Falls back to ``flash_mla`` only when that
build is unavailable.

Used by MLA prefill and chunked-context paths where Q, K, V are dense
``[total_tokens, num_heads, head_dim]`` tensors (no paged cache lookup,
no ``block_table``).  Supports ``return_softmax_lse`` for MLA chunked
prefix merging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fa_utils import FA_VERSION, VLLM_FA_AVAILABLE, flash_attn_varlen_func

_fa2_varlen_func = None
_flashmla_varlen_func = None
if not VLLM_FA_AVAILABLE:  # pragma: no cover - CPU-only fallback
    try:
        from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
    except ImportError:
        # vLLM vendors FlashMLA; fall back to the standalone ``flash_mla``
        # package when the vendored copy is unavailable.
        try:
            from vllm.third_party.flashmla.flash_mla_interface import (
                flash_attn_varlen_func as _flashmla_varlen_func,
            )
        except ImportError:  # pragma: no cover
            from flash_mla import (  # type: ignore[no-redef]
                flash_attn_varlen_func as _flashmla_varlen_func,
            )


class FlashAttnVarlen(nn.Module):
    """Variable-length Flash Attention without paged KV cache lookup."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        if VLLM_FA_AVAILABLE:
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                return_softmax_lse=return_softmax_lse,
                fa_version=FA_VERSION,
            )
        fn = _fa2_varlen_func if _fa2_varlen_func is not None else _flashmla_varlen_func
        kwargs = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        if return_softmax_lse:
            kwargs["return_softmax_lse"] = return_softmax_lse
        return fn(q, k, v, **kwargs)
