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

We delegate to vLLM's own ``get_flash_attn_version()`` when it imports so the
two can never drift, and only fall back to the local reimplementation if that
module is unavailable.
"""

from __future__ import annotations

import torch

__all__ = [
    "VLLM_FA_AVAILABLE",
    "FA_VERSION",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "fa_supports_head_size",
]

VLLM_FA_AVAILABLE = False
FA_VERSION: int | None = None
flash_attn_varlen_func = None
get_scheduler_metadata = None

_is_fa_version_supported = None

try:
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        flash_attn_varlen_func as _vllm_flash_attn_varlen_func,
        get_scheduler_metadata as _vllm_get_scheduler_metadata,
        is_fa_version_supported as _is_fa_version_supported,
    )

    VLLM_FA_AVAILABLE = torch.cuda.is_available()
    if VLLM_FA_AVAILABLE:
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
        flash_attn_varlen_func = torch._dynamo.disable(
            _vllm_flash_attn_varlen_func,
        )
        get_scheduler_metadata = _vllm_get_scheduler_metadata
except ImportError:  # pragma: no cover - vLLM always present in this env
    pass


def _local_get_flash_attn_version() -> int | None:
    """Reimplementation of vLLM's device -> fa_version mapping."""
    if not VLLM_FA_AVAILABLE or _is_fa_version_supported is None:
        return None
    major = torch.cuda.get_device_capability()[0]
    if major == 9 and _is_fa_version_supported(3):
        return 3
    if major >= 10 and _is_fa_version_supported(4):
        return 4
    return 2


if VLLM_FA_AVAILABLE:
    try:
        from vllm.v1.attention.backends.fa_utils import (
            get_flash_attn_version as _vllm_get_flash_attn_version,
        )

        FA_VERSION = _vllm_get_flash_attn_version()
    except Exception:
        FA_VERSION = _local_get_flash_attn_version()
    if FA_VERSION is None:
        FA_VERSION = _local_get_flash_attn_version()


def fa_supports_head_size(head_size: int) -> bool:
    """Mirror ``FlashAttentionBackend.supports_head_size`` from vLLM 0.26."""
    if head_size % 8 != 0:
        return False
    if head_size <= 256:
        return True
    if _is_fa_version_supported is not None and _is_fa_version_supported(4):
        return head_size <= 512
    return False
