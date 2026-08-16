"""FlashAttention version selection, mirroring vLLM's ``fa_utils``.

fastkernels calls the *same* FlashAttention build vLLM does --
``vllm.vllm_flash_attn`` -- rather than the standalone PyPI ``flash_attn``
package.  Two reasons:

1. **Numerical alignment.**  A vLLM-aligned benchmark has to run the kernel
   vLLM would run.  ``vllm.vllm_flash_attn`` bundles FA2, FA3 (Hopper) and
   FA4 (CuTeDSL, Blackwell); which one is picked depends on the device, and
   ``get_flash_attn_version`` reproduces that choice.

2. **Paged-KV support.**  Upstream ``flash_attn`` >= 2.8 rejects any paged
   KV cache whose page size is not a multiple of 256
   (``"Paged KV cache block size must be divisible by 256"``).  vLLM's build
   accepts ``MultipleOf(16)``, which is what the engine's block size is.
   Routing paged attention through the PyPI package therefore fails outright
   on every backend config fastkernels uses.

The version choice follows ``vllm/v1/attention/backends/fa_utils.py``:

    SM90  + FA3 available -> 3
    SM100 + FA4 available -> 4
    otherwise             -> 2

We delegate to vLLM's own ``get_flash_attn_version()`` so the two can never
drift.
"""

from __future__ import annotations

import torch
from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
    flash_attn_varlen_func as _vllm_flash_attn_varlen_func,
    get_scheduler_metadata as _vllm_get_scheduler_metadata,
    is_fa_version_supported as _is_fa_version_supported,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version as _vllm_get_flash_attn_version,
)

__all__ = [
    "VLLM_FA_AVAILABLE",
    "FA_VERSION",
    "FA3_CUDA_GRAPH_MAX_NUM_SPLITS",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "fa3_scheduler_metadata",
    "fa3_scheduler_metadata_size",
    "fa_supports_head_size",
    "group_fa_decode_ops",
    "refresh_fa3_decode_schedule",
]

VLLM_FA_AVAILABLE = torch.cuda.is_available()

# Hide the kernel from Dynamo.  vLLM reaches FlashAttention only
# through ``direct_register_custom_op`` boundaries
# (``unified_attention_with_output`` et al), so Dynamo never traces
# into it.  fastkernels calls this function *inline* from encoder
# attention (``encoder_attention.FlashAttnVarlen``), which torch.compile
# does trace -- and FA4's CuTeDSL launcher is pure Python that rebuilds
# a ``_kwargs_wrapper`` closure per call. Dynamo guards on that closure's
# code object, so every call missed its guard and recompiled: the
# embedding harness hit ``recompile_limit (8)`` and each single-request
# encode then cost ~28.5 s versus vLLM's 0.003 s.
# ``disable`` restores the opaque-kernel semantics the PyPI flash_attn
# C++ extension had for free, and is a no-op for the LLM path where
# attention already sits behind ``fastkernels::unified_attention``.
flash_attn_varlen_func = torch._dynamo.disable(_vllm_flash_attn_varlen_func)
get_scheduler_metadata = _vllm_get_scheduler_metadata

FA_VERSION: int | None = (
    _vllm_get_flash_attn_version() if VLLM_FA_AVAILABLE else None
)

# vLLM ``AttentionConfig.flash_attn_max_num_splits_for_cuda_graph``.
# FA3's auto split heuristic is not CUDA-graph compatible; vLLM pins this
# upper bound so capture pre-allocates large enough split-KV scratch.
FA3_CUDA_GRAPH_MAX_NUM_SPLITS = 32


def fa3_scheduler_metadata_size(batch_size: int) -> int:
    """Bytes-as-int32s of an FA3 ``scheduler_metadata`` tensor.

    ``1 + round_up(batch_size, 4) * 4`` -- the +1 is the tile-count
    semaphore; the 4 slots per rounded batch element are prepare_varlen,
    dynamic_split, sort_batches, head_swizzle.  Matches
    ``FlashAttentionMetadataBuilder`` in vLLM's ``flash_attn.py``.
    """
    return 1 + ((batch_size + 3) // 4 * 4) * 4


def fa3_scheduler_metadata(**kwargs):
    """Build FA3 tile-scheduler metadata, or None when FA3 is not in use.

    Hopper FA3 illegal-memory-accesses on long / large-batch paged attention
    unless this tensor is passed through to ``flash_attn_varlen_func``.
    vLLM always supplies it (``aot_schedule = get_flash_attn_version() == 3``).
    """
    if FA_VERSION != 3:
        return None
    return get_scheduler_metadata(**kwargs)


def group_fa_decode_ops(ops):
    """Group FlashAttnDecode modules that share one FA3 schedule.

    vLLM builds one ``FlashAttentionMetadata`` per KV-cache group
    (same heads / window / page size).  Identical layers can share the
    persistent scheduler buffer and a single ``get_scheduler_metadata``.
    """
    groups: dict[tuple, list] = {}
    for op in ops:
        key = (
            op.num_heads,
            op.num_kv_heads,
            op.head_dim,
            tuple(getattr(op, "_window_size", (-1, -1))),
            getattr(op, "page_size", None),
        )
        groups.setdefault(key, []).append(op)
    return list(groups.values())


def refresh_fa3_decode_schedule(
    groups,
    context_lens: torch.Tensor,
    max_seqlen_k: int,
    qkv_dtype: torch.dtype,
) -> None:
    """Rewrite FA3 tile-scheduler metadata for this decode step.

    Must run OUTSIDE CUDA graph capture/replay, matching vLLM's
    ``FlashAttentionMetadataBuilder``.  ``context_lens`` is the [B]
    decode lengths *including* the padded tail the captured graph
    covers.  Layers in a group share one persistent buffer so every
    captured kernel reads the updated schedule on replay.
    """
    if FA_VERSION != 3 or not groups:
        return
    with torch.inference_mode():
        for group in groups:
            lead = group[0]
            lead.update_scheduler_metadata(
                context_lens,
                max_seqlen_k,
                qkv_dtype=qkv_dtype,
                window_size=getattr(lead, "_window_size", (-1, -1)),
            )
            for op in group[1:]:
                op._sched_buf = lead._sched_buf
                op._sched_meta = lead._sched_meta
                op._sched_meta_batch = lead._sched_meta_batch


def fa_supports_head_size(head_size: int) -> bool:
    """Mirror ``FlashAttentionBackend.supports_head_size`` from vLLM 0.26."""
    if head_size % 8 != 0:
        return False
    if head_size <= 256:
        return True
    if _is_fa_version_supported(4):
        return head_size <= 512
    return False
