"""Flash attention decode kernel (with paged KV cache).

Uses the unified ``flash_attn_varlen_func`` interface from vLLM's bundled
FlashAttention build, at the version vLLM would select for this device
(FA3 on Hopper, FA4 on Blackwell, FA2 otherwise) -- see :mod:`fa_utils`.
That is also the only build whose paged-KV path accepts the engine's block
size; upstream ``flash_attn`` requires page sizes divisible by 256.
"""

import torch
import torch.nn as nn

from .fa_utils import FA_VERSION, flash_attn_varlen_func


class FlashAttnDecode(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._cu_seqlens_q = None

    def _get_cu_seqlens_q(self, n: int, device: torch.device) -> torch.Tensor:
        needed = n + 1
        if self._cu_seqlens_q is None or self._cu_seqlens_q.numel() < needed:
            self._cu_seqlens_q = torch.arange(
                needed, dtype=torch.int32, device=device,
            )
        return self._cu_seqlens_q[:needed]

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        max_seq_len = kwargs.pop("max_seq_len", None)
        block_table = kwargs.pop("block_table", None)
        softmax_scale = kwargs.pop("softmax_scale", None)
        kwargs.pop("causal", None)

        n = q.shape[0]
        cu_seqlens_q = self._get_cu_seqlens_q(n, q.device)
        if max_seq_len is not None:
            max_seqlen_k = max_seq_len
        else:
            max_seqlen_k = int(cache_seqlens.max().item()) if cache_seqlens.numel() > 0 else 0

        fa_kw = dict(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=block_table,
            fa_version=FA_VERSION,
        )
        fa_kw.update(kwargs)
        return flash_attn_varlen_func(**fa_kw)
