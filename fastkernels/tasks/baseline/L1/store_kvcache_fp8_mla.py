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

from .csrc import _C

_KV_C_DIM = 512
_K_PE_DIM = 64
_FP8_BYTES_PER_TOKEN = 656
_BF16_ELEMS_PER_TOKEN = _KV_C_DIM + _K_PE_DIM  # 576


class StoreKVCacheFP8MLA(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into MLA paged cache.

    Wraps vendored ``_C.concat_and_cache_mla``. Dispatches on
    ``kv_cache_dtype``:

    * ``"auto"``: expects a BF16 cache of shape
      ``[num_blocks, block_size, 576]``; the kernel writes the
      concatenation of ``kv_c_normed`` (512) and ``k_pe`` (64) directly.
    * ``"fp8_ds_mla"``: expects a uint8 cache of shape
      ``[num_blocks, block_size, 656]``; the kernel fuses per-block
      UE8M0 FP8 quantization of ``kv_c_normed`` with BF16 ``k_pe`` storage.
    * ``"fp8_e4m3"``: expects a ``float8_e4m3fn`` cache of shape
      ``[num_blocks, block_size, 576]``; the kernel divides both halves by
      ``k_scale`` and casts to fp8. This is the layout vLLM's
      FLASHINFER_MLA_SPARSE backend uses -- plain per-tensor fp8, no block
      scales -- and it is NOT interchangeable with ``fp8_ds_mla``.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 — compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 — RoPE key component.
        kv_cache: ``[num_blocks, block_size, 576|656]`` (BF16 / fp8 / uint8).
        slot_mapping: ``[N]`` int64 — linear slot index per token (``-1`` skips).
    """

    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"StoreKVCacheFP8MLA: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ``k_scale`` is the per-tensor dequant scale the kernel divides by on
        # the ``fp8_e4m3`` path (it is ignored for ``auto`` and for the
        # block-scaled ``fp8_ds_mla`` layout). vLLM initialises ``layer._k_scale``
        # to 1.0 and only overwrites it from a checkpoint's calibration scales,
        # which nvidia/GLM-5.2-NVFP4 does not ship -- so ONE, not zero. A zero
        # here silently turned every stored KV element into inf/nan.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        k_pe_2d = k_pe.reshape(k_pe.shape[0], -1)
        _C.concat_and_cache_mla(
            kv_c_normed, k_pe_2d, kv_cache, slot_mapping,
            self.kv_cache_dtype, self._k_scale,
        )


class GatherKVCacheFP8MLA(nn.Module):
    """Gather and upconvert KV from FP8 MLA paged cache to BF16.

    Wraps vendored ``_C.cp_gather_and_upconvert_fp8_kv_cache``
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
        # ``seq_lens`` is unused by the kernel; lengths are implied by
        # ``workspace_starts`` + ``workspace.size(0)``. Kept in the Module API
        # for call-site compatibility.
        del seq_lens
        _C.cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache, workspace, block_table, workspace_starts, num_seqs, None,
        )


class GatherAndDequantKVCacheMLA(nn.Module):
    """Gather MLA KV cache into a BF16 workspace using the
    ``gather_and_maybe_dequant_cache`` kernel (vLLM's chunked-context helper).

    Required arguments match the kernel's signature:
        ``kv_cache``: ``[num_blocks, block_size, 576]`` BF16 (``"auto"``) or
                      ``[num_blocks, block_size, 656]`` uint8 (``fp8_ds_mla``).
        ``workspace``: ``[total_tokens, 576]`` BF16 output buffer.
        ``block_table``: ``[num_seqs, max_blocks]`` int32.
        ``cu_seq_lens``: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        ``token_to_seq``: ``[total_tokens]`` int32 mapping.
        ``total_tokens``: scalar int.
        ``workspace_starts``: ``[num_seqs]`` int32 — starting workspace row
                             per sequence (for chunked context gathers).

    ``kv_cache_dtype`` selects the source layout and must match the cache the
    owning ``MLAAttention`` allocated; vLLM likewise forwards its own
    ``self.kv_cache_dtype`` here, and passing ``"fp8_ds_mla"`` for a BF16
    cache reinterprets the bytes and silently corrupts the gathered context.
    """

    def __init__(self, kv_cache_dtype: str = "fp8_ds_mla"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"GatherAndDequantKVCacheMLA: unsupported "
            f"kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ONE, not zero: on the ``fp8_e4m3`` path the kernel MULTIPLIES the
        # gathered fp8 values by this scale to dequantize (vLLM passes
        # ``layer._k_scale``, default 1.0). Zero would blank the gathered
        # context. Ignored for ``auto`` and for the block-scaled ``fp8_ds_mla``.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
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
        _C.gather_and_maybe_dequant_cache(
            kv_cache, workspace,
            block_table, cu_seq_lens, token_to_seq,
            total_tokens,
            self.kv_cache_dtype,
            self._k_scale,
            workspace_starts,
        )
