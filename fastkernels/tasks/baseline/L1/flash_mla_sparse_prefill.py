"""FlashMLA sparse prefill for DSA (DeepSeek Sparse Attention)."""

from __future__ import annotations

import torch
import torch.nn as nn

# vLLM vendors FlashMLA; fall back to the standalone package if not present.
try:
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_sparse_fwd
except ImportError:  # pragma: no cover
    from flash_mla import flash_mla_sparse_fwd  # type: ignore[no-redef]


class FlashMLASparsePrefill(nn.Module):
    """Wraps flash_mla.flash_mla_sparse_fwd for sparse BF16 prefill.

    Used when prefill has sparse indices (DSA). The workspace must
    already contain BF16 KV data gathered from the FP8 cache.
    """

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        d_v: int = 512,
    ) -> torch.Tensor:
        return flash_mla_sparse_fwd(q, kv, indices, softmax_scale, d_v=d_v)
