"""TRTLLM-gen paged attention decode kernel (via FlashInfer, Blackwell only).

Accepts the same interface as FlashAttnDecode so that LlamaAttention can
dispatch to either backend without branch logic.
"""

import torch
import torch.nn as nn
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


class TRTLLMDecode(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        if workspace is None:
            workspace = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self._workspace = workspace

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, **kwargs):
        if max_seq_len is None:
            max_seq_len = int(cache_seqlens.max().item())
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            bmm2_scale=1.0,
            kv_layout="HND",
        )
