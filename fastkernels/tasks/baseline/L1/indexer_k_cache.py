"""Indexer K cache store and gather for DSA (132 bytes/token).

Uses vLLM's CUDA kernels for high-performance FP8 quantization and cache
operations, matching the exact semantics of the vLLM sparse attention indexer.

Cache format per token (132 bytes):

  * ``[0:128]`` — K as ``float8_e4m3fn``.
  * ``[128:132]`` — one float32 UE8M0 scale for the full head.

Cache tensor shape: ``[num_blocks, block_size, 132]`` with ``dtype=torch.uint8``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm._C  # noqa: F401  — registers torch.ops._C_cache_ops

_HEAD_DIM = 128
_SCALE_BYTES = 4
_BYTES_PER_TOKEN = _HEAD_DIM + _SCALE_BYTES
_QUANT_BLOCK_SIZE = 128


class IndexerKCacheStore(nn.Module):
    """Quantize K to FP8 and store in indexer paged cache via CUDA kernel.

    Wraps ``torch.ops._C_cache_ops.indexer_k_quant_and_cache`` which fuses
    FP8 quantization (UE8M0 scale) with paged cache insertion.

    Args:
        k: ``[N, head_dim]`` BF16 — indexer key vectors.
        kv_cache: ``[num_blocks, block_size, 132]`` uint8.
        slot_mapping: ``[N]`` int32 — linear slot index per token.
    """

    def forward(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        torch.ops._C_cache_ops.indexer_k_quant_and_cache(
            k, kv_cache, slot_mapping, _QUANT_BLOCK_SIZE, "ue8m0",
        )


class IndexerKCacheGather(nn.Module):
    """Gather K from indexer paged cache in FP8 form via CUDA kernel.

    Wraps ``torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache`` which
    gathers FP8 keys and their float32 scales from a paged cache.

    Returns:
        ``k_fp8``: ``[total_tokens, head_dim]`` float8_e4m3fn.
        ``k_scale``: ``[total_tokens, 4]`` uint8 (float32 scale viewed as bytes).
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_tokens = int(cu_seq_lens[-1].item())
        device = kv_cache.device

        k_fp8 = torch.empty(
            total_tokens, _HEAD_DIM, dtype=torch.float8_e4m3fn, device=device,
        )
        k_scale = torch.empty(
            total_tokens, _SCALE_BYTES, dtype=torch.uint8, device=device,
        )

        torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
            kv_cache, k_fp8, k_scale, block_table, cu_seq_lens,
        )

        return k_fp8, k_scale
