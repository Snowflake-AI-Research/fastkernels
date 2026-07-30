"""TRTLLM-gen paged attention decode kernel (via FlashInfer, Blackwell only).

Accepts the same interface as FlashAttnDecode so that LlamaAttention can
dispatch to either backend without branch logic.
"""

import torch
import torch.nn as nn
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


def prime_trtllm_sinks(module: nn.Module, sinks: torch.Tensor | None) -> None:
    """Materialize the FP32 attention-sink copy the trtllm-gen kernels need.

    ``trtllm_batch_decode_with_kv_cache`` /
    ``trtllm_batch_context_with_kv_cache`` hard-check
    ``attention_sinks.dtype == float32``, while the FlashAttention build vLLM
    bundles asserts the opposite for the same weights
    (``learnable_sink must be bfloat16``).  So the conversion cannot live on the
    layer -- only the op knows which kernel it is about to call.  vLLM does the
    same conversion once per layer in
    ``FlashInferImpl.process_weights_after_loading``; call this from the owning
    attention layer's post-load hook so the copy never lands inside a forward
    or a CUDA-graph capture.
    """
    if sinks is None:
        module._sinks_fp32 = None
    elif sinks.dtype == torch.float32:
        module._sinks_fp32 = sinks
    else:
        module._sinks_fp32 = sinks.detach().to(torch.float32)
    module._sinks_src = sinks


def trtllm_sinks(module: nn.Module, s_aux: torch.Tensor | None):
    """Return the FP32 view of ``s_aux``, priming the cache if needed."""
    if s_aux is None or s_aux.dtype == torch.float32:
        return s_aux
    if module._sinks_fp32 is None or module._sinks_src is not s_aux:
        prime_trtllm_sinks(module, s_aux)
    return module._sinks_fp32


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
        self._sinks_fp32: torch.Tensor | None = None
        self._sinks_src: torch.Tensor | None = None

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        prime_trtllm_sinks(self, sinks)

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, s_aux=None, window_size=None, **kwargs):
        if max_seq_len is None:
            max_seq_len = int(cache_seqlens.max().item())
        # trtllm-gen requires a contiguous query: with a batched (multi-request)
        # decode, the query view is non-contiguous and the TMA load reads later
        # rows at the wrong stride -> only row 0 is correct, the rest are garbage.
        # vLLM's FlashInfer backend and our own TRTLLMPrefill both do this; the
        # decode path was missing it.
        q = q.contiguous()
        # block_tables / seq_lens MUST be contiguous: the trtllm-gen kernel
        # reads the page table assuming a dense [batch, max_pages] row-major
        # layout. The engine's eager/CUDA-graph decode buffers hand us a column
        # slice (``_eager_block_tables[:n, :bt_cols]``) whose row stride is the
        # full ``max_num_blocks``, not ``bt_cols`` -> every row > 0 would read
        # its page ids from the wrong offset (garbage pages), so only row 0
        # stayed correct and all other sequences in the batch were corrupted.
        # vLLM likewise asserts is_strictly_contiguous(block_tables/seq_lens).
        block_table = block_table.contiguous()
        if cache_seqlens is not None:
            cache_seqlens = cache_seqlens.contiguous()
        # Attention sinks and the sliding window must be forwarded explicitly.
        # The caller names them ``s_aux`` / ``window_size`` (the FlashAttention
        # spelling); trtllm-gen calls them ``sinks`` / ``window_left``. Letting
        # them fall into **kwargs silently dropped both, which is a *numerical*
        # bug, not a crash: gpt-oss-120b (sinks + alternating sliding window)
        # scored 0.8 of 385 matching tokens against vLLM. vLLM passes both here
        # (flashinfer.py: window_left=self.window_left, sinks=self.sinks).
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            bmm2_scale=1.0,
            window_left=(
                window_size[0] if window_size is not None
                and window_size[0] >= 0 else -1
            ),
            sinks=trtllm_sinks(self, s_aux),
            kv_layout="HND",
        )
