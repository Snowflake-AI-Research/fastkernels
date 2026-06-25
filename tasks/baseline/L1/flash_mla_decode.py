"""FlashMLA decode kernels for MLA (Multi-head Latent Attention).

Mirrors vLLM's two distinct decode entry points:

* ``flash_mla_with_kvcache`` — generic decode kernel. Used for BF16 KV cache
  and for *sparse* FP8 decode (which routes FP8 through the generic kernel
  with ``is_fp8_kvcache=True`` plus ``indices``).
* ``flash_mla_with_kvcache_fp8`` — dedicated dense FP8 decode kernel. Takes
  ``descale_q`` / ``descale_k`` per-layer scales and a ``num_splits`` tensor
  produced by ``get_mla_metadata_dense_fp8``.

vLLM's ``FlashMLAImpl.forward_mqa`` switches between the two on
``self.kv_cache_dtype.startswith("fp8")``; we replicate that branching at the
caller level (``MLAAttention._forward_dense_decode`` / ``_forward_mixed``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# vLLM ships a fully vendored copy of FlashMLA under
# ``vllm.third_party.flashmla.flash_mla_interface`` together with its
# compiled ``vllm._flashmla_C`` / ``vllm._flashmla_extension_C`` kernels,
# so fastkernels does not need to build FlashMLA separately.  The standalone
# ``flash_mla`` package remains supported as a fallback for environments
# that predate the vendored copy.
try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
except ImportError:  # pragma: no cover
    from flash_mla import (  # type: ignore[no-redef]
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

# FP8-specific dense decode entry points (vLLM-only). Falls back to ``None``
# when the vendored kernels are not available, which forces the dense FP8
# path to use the generic ``flash_mla_with_kvcache(..., is_fp8_kvcache=True)``
# fallback (matches behaviour on hardware without the FP8-specialized kernel).
try:
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache_fp8,
        get_mla_metadata_dense_fp8,
    )
except ImportError:  # pragma: no cover
    flash_mla_with_kvcache_fp8 = None  # type: ignore[assignment]
    get_mla_metadata_dense_fp8 = None  # type: ignore[assignment]


class FlashMLADecode(nn.Module):
    """Wraps ``flash_mla_with_kvcache`` for paged MLA decode.

    Used for:
    - BF16 dense decode (``is_fp8_kvcache=False``)
    - Sparse FP8 decode with ``indices`` (DSA path)

    Dense FP8 decode goes through :class:`FlashMLADecodeFP8` instead, which
    matches vLLM's ``flash_mla_with_kvcache_fp8`` entry point with
    ``descale_q`` / ``descale_k`` and ``num_splits``.
    """

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        head_dim_v: int,
        tile_scheduler_metadata: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        is_fp8_kvcache: bool = False,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return flash_mla_with_kvcache(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            softmax_scale=softmax_scale,
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
        )


class FlashMLADecodeFP8(nn.Module):
    """Dense FP8 decode wrapper matching vLLM's ``flash_mla_with_kvcache_fp8``.

    Requires ``descale_q`` / ``descale_k`` (per-layer Q/K dequantization
    scales) and a ``num_splits`` tensor produced by
    :class:`FlashMLAGetMetadataDenseFP8`.
    """

    available: bool = flash_mla_with_kvcache_fp8 is not None

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        head_dim_v: int,
        tile_scheduler_metadata: torch.Tensor,
        num_splits: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        descale_q: torch.Tensor | None = None,
        descale_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert flash_mla_with_kvcache_fp8 is not None, (
            "flash_mla_with_kvcache_fp8 not available — "
            "vLLM build must include _flashmla_extension_C"
        )
        return flash_mla_with_kvcache_fp8(
            q=q,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            descale_q=descale_q,
            descale_k=descale_k,
        )


class FlashMLAGetMetadata(nn.Module):
    def forward(
        self,
        cache_seqlens: torch.Tensor,
        num_q_tokens_per_head_k: int,
        num_heads_k: int = 1,
        topk: int | None = None,
        num_heads_q: int | None = None,
        is_fp8_kvcache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict = {}
        if topk is not None:
            kwargs["topk"] = topk
        if num_heads_q is not None:
            kwargs["num_heads_q"] = num_heads_q
            kwargs["num_heads_k"] = num_heads_k
        if is_fp8_kvcache:
            kwargs["is_fp8_kvcache"] = is_fp8_kvcache
        if kwargs:
            return get_mla_metadata(
                cache_seqlens=cache_seqlens,
                num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                **kwargs,
            )
        return get_mla_metadata(cache_seqlens, num_q_tokens_per_head_k, num_heads_k)


class FlashMLAGetMetadataDenseFP8(nn.Module):
    """Wraps ``get_mla_metadata_dense_fp8`` for the FP8 dense decode kernel.

    Returns ``(tile_scheduler_metadata, num_splits)`` which both must be fed
    into :class:`FlashMLADecodeFP8`. Matches the vLLM call site in
    ``vllm/v1/attention/backends/mla/flashmla.py:171-178``.
    """

    available: bool = get_mla_metadata_dense_fp8 is not None

    def forward(
        self,
        cache_seqlens: torch.Tensor,
        num_q_tokens_per_head_k: int,
        num_heads_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert get_mla_metadata_dense_fp8 is not None, (
            "get_mla_metadata_dense_fp8 not available — "
            "vLLM build must include _flashmla_extension_C"
        )
        return get_mla_metadata_dense_fp8(
            cache_seqlens, num_q_tokens_per_head_k, num_heads_k,
        )
