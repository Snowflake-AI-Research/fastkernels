"""FP8 MQA logits for DSA indexer via DeepGEMM.

Uses DeepGEMM's UNIFIED FP8/FP4 kernels ``fp8_fp4_mqa_logits`` /
``fp8_fp4_paged_mqa_logits`` — the EXACT functions vLLM's DSA indexer calls
(``vllm.utils.deep_gemm``). deep_gemm 2.5.0 also ships the legacy
``fp8_mqa_logits`` / ``fp8_paged_mqa_logits``, but they are a DIFFERENT kernel
that produces bit-different index scores, which flips the selected top-2048 set
at seq>2048. On the FP8 path the query is passed as ``(q_fp8, None)`` (the
per-token Q scale is folded into ``weights`` upstream), matching vLLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import deep_gemm


__targets__ = ["Fp8MQALogits"]


class Fp8MQALogits(nn.Module):
    """FP8 MQA logits for the DSA indexer (vLLM's unified fp8/fp4 kernels).

    Prefill path: :func:`deep_gemm.fp8_fp4_mqa_logits`.
    Decode path: :func:`deep_gemm.fp8_fp4_paged_mqa_logits`.
    """

    def forward_prefill(
        self,
        q_fp8: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        return deep_gemm.fp8_fp4_mqa_logits(
            (q_fp8, None),   # FP8 Q: scale folded into weights (matches vLLM)
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=False,
        )

    def forward_decode(
        self,
        q_fp8: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        schedule_metadata: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp8, None),   # FP8 Q: scale folded into weights (matches vLLM)
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_context_len,
            clean_logits=False,
        )


class Fp8PagedMQALogitsMetadata(nn.Module):
    """Scheduling metadata for paged MQA logits (DeepGEMM)."""

    def forward(
        self,
        context_lens: torch.Tensor,
        block_size: int,
        num_sms: int,
    ) -> torch.Tensor:
        return deep_gemm.get_paged_mqa_logits_metadata(
            context_lens,
            block_size,
            num_sms,
        )
