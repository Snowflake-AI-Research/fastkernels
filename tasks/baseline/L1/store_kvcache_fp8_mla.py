"""MLA KV cache store and gather.

Supports two cache layouts via ``kv_cache_dtype``:

* ``"auto"`` (default, matches vLLM): BF16 KV cache with shape
  ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``
  (e.g. 576 BF16 elements = 1152 bytes/token for DeepSeek-V3.2).
  vLLM's ``concat_and_cache_mla`` with ``kv_cache_dtype="auto"`` writes
  ``kv_c_normed`` and ``k_pe`` directly as BF16 — no quantization.
* ``"fp8_ds_mla"``: FP8 KV cache (656 bytes/token):

  * ``[0:512]`` — ``kv_c_normed`` as FP8 (``float8_e4m3fn``).
  * ``[512:528]`` — four per-group FP32 UE8M0 scales (128 dims per group).
  * ``[528:656]`` — ``k_pe`` as 64 ``bfloat16`` values (128 bytes).

  Cache tensor shape: ``[num_blocks, block_size, 656]`` with ``dtype=torch.uint8``.

The default is BF16 to match vLLM's stock behaviour (``kv_cache_dtype=auto``
on DeepSeek-V3.2 selects BF16 KV cache). Use the ``FASTKERNELS_KV_CACHE_DTYPE``
env var to force ``fp8_ds_mla`` for extra memory savings at the cost of
numerical drift vs. vLLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm._C  # noqa: F401  — registers torch.ops._C_cache_ops

_KV_C_DIM = 512
_K_PE_DIM = 64
_FP8_BYTES_PER_TOKEN = 656
_BF16_ELEMS_PER_TOKEN = _KV_C_DIM + _K_PE_DIM  # 576

_HAS_GATHER_AND_DEQUANT = (
    hasattr(torch.ops, "_C_cache_ops")
    and hasattr(torch.ops._C_cache_ops, "gather_and_maybe_dequant_cache")
)


class StoreKVCacheFP8MLA(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into MLA paged cache.

    Wraps ``torch.ops._C_cache_ops.concat_and_cache_mla``. Dispatches on
    ``kv_cache_dtype``:

    * ``"auto"``: expects a BF16 cache of shape
      ``[num_blocks, block_size, 576]``; the kernel writes the
      concatenation of ``kv_c_normed`` (512) and ``k_pe`` (64) directly.
    * ``"fp8_ds_mla"``: expects a uint8 cache of shape
      ``[num_blocks, block_size, 656]``; the kernel fuses per-block
      UE8M0 FP8 quantization of ``kv_c_normed`` with BF16 ``k_pe`` storage.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 — compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 — RoPE key component.
        kv_cache: ``[num_blocks, block_size, 576|656]`` (BF16 or uint8).
        slot_mapping: ``[N]`` int64 — linear slot index per token (``-1`` skips).
    """

    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla"), (
            f"StoreKVCacheFP8MLA: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.register_buffer(
            "_k_scale", torch.zeros(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        k_pe_2d = k_pe.reshape(k_pe.shape[0], -1)
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c_normed, k_pe_2d, kv_cache, slot_mapping,
            self.kv_cache_dtype, self._k_scale,
        )


class GatherKVCacheFP8MLA(nn.Module):
    """Gather and upconvert KV from FP8 MLA paged cache to BF16.

    Wraps ``torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache``
    which gathers FP8-quantized kv_c_normed and BF16 k_pe from paged cache,
    dequantizes the FP8 portion, and writes the result as a contiguous
    BF16 workspace tensor.

    Returns:
        ``workspace``: ``[total_tokens, 576]`` BF16 — dequantized kv_c_normed
        (512 dims) concatenated with k_pe (64 dims).
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        num_seqs: int,
        workspace: torch.Tensor,
    ) -> None:
        torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache, workspace, block_table, seq_lens,
            workspace_starts, num_seqs,
        )


class GatherAndDequantKVCacheMLA(nn.Module):
    """Gather FP8 MLA KV cache into a BF16 workspace using the
    ``gather_and_maybe_dequant_cache`` kernel (vLLM's chunked-context helper).

    Required arguments match the kernel's signature:
        ``kv_cache``: ``[num_blocks, block_size, 656]`` uint8.
        ``workspace``: ``[total_tokens, 576]`` BF16 output buffer.
        ``block_table``: ``[num_seqs, max_blocks]`` int32.
        ``cu_seq_lens``: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        ``token_to_seq``: ``[total_tokens]`` int32 mapping.
        ``total_tokens``: scalar int.
        ``workspace_starts``: ``[num_seqs]`` int32 — starting workspace row
                             per sequence (for chunked context gathers).

    Raises ``RuntimeError`` if the kernel is unavailable; callers should
    check :pyattr:`available` and fall back to :class:`GatherKVCacheFP8MLA`.
    """

    available: bool = _HAS_GATHER_AND_DEQUANT

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "_k_scale", torch.zeros(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        total_tokens: int,
        workspace_starts: torch.Tensor,
    ) -> None:
        if not _HAS_GATHER_AND_DEQUANT:
            raise RuntimeError(
                "torch.ops._C_cache_ops.gather_and_maybe_dequant_cache is "
                "unavailable; use GatherKVCacheFP8MLA instead.",
            )
        torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
            kv_cache, workspace,
            block_table, cu_seq_lens, token_to_seq,
            total_tokens,
            "fp8_ds_mla",
            self._k_scale,
            workspace_starts,
        )
