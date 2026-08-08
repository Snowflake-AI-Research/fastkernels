"""FlashInfer trtllm-gen MLA decode kernel (Blackwell).

FlashMLA's dense decode kernel is **SM90a-only**::

    RuntimeError: dense_attn_decode_interface,
    flashmla-src/csrc/api/dense_decode.h:29,
    Dense decode MLA is only supported on SM90a architecture

so on Blackwell it cannot run at all. vLLM selects ``FLASHINFER_MLA`` there
(logged as ``Using FLASHINFER_MLA attention backend`` /
``Using HND KV cache layout for FLASHINFER_MLA``) and dispatches decode through
``trtllm_batch_decode_with_kv_cache_mla``. This module wraps that same entry
point with the argument shape our :class:`MLAAttention` decode path already
produces, so the two run the same kernel.

Mirrors ``FlashInferMLAImpl.forward_mqa``
(``vllm/v1/attention/backends/mla/flashinfer_mla.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla


def flashinfer_mla_decode_supported() -> bool:
    """True when the trtllm-gen MLA decode kernel can run on this device."""
    if not torch.cuda.is_available():
        return False
    # trtllm-gen MLA is a Blackwell (SM100) path.
    return torch.cuda.get_device_capability()[0] >= 10


class FlashInferMLADecode(nn.Module):
    """trtllm-gen MLA decode over a paged latent KV cache.

    Parameters mirror the FlashMLA decode op this stands in for, so the
    call site does not need to branch beyond choosing the module.
    """

    # vLLM keeps one workspace per (return_lse) variant; a single shared
    # buffer is enough here since we never request the LSE.
    _WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self._workspace = workspace

    @property
    def available(self) -> bool:
        return flashinfer_mla_decode_supported()

    def ensure_workspaces(self, device: torch.device) -> None:
        """Materialize the trtllm-gen MLA decode workspace before graph capture.

        See ``TopKPerRow.ensure_workspaces``: a workspace first allocated inside
        a capture region belongs to that graph's private pool, and later graphs
        replaying against it fault.
        """
        self._get_workspace(device)

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.zeros(
                self._WORKSPACE_BYTES, dtype=torch.uint8, device=device,
            )
        return self._workspace

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        max_seq_len: int,
        bmm2_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        q : ``[num_decodes, q_len, num_heads, qk_head_dim]``
            Same layout the FlashMLA decode op receives (``q.unsqueeze(1)``).
        kv_cache : ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``
            Paged latent cache. vLLM passes ``kv_c_and_k_pe_cache.unsqueeze(1)``,
            i.e. a singleton head dim, which we add here if absent.
        block_table, cache_seqlens
            Per-request page table and context lengths.

        Returns ``(out, None)`` to match the FlashMLA op's ``(out, lse)``.
        """
        if kv_cache.dim() == 3:
            kv_cache = kv_cache.unsqueeze(1)

        # trtllm-gen walks the page table in 128-token strides, so it rejects a
        # width that is not a multiple of ceil(128 / page_size). Callers size
        # the table from max_model_len, so pad here as a backstop; the extra
        # columns are never read (the walk is bounded by seq_lens).
        page_size = kv_cache.shape[2]
        gran = max(1, -(-128 // page_size))
        if block_table.shape[-1] % gran:
            pad = gran - block_table.shape[-1] % gran
            block_table = torch.nn.functional.pad(block_table, (0, pad))

        # trtllm-gen reads the page table and seq lens as dense row-major
        # tensors; a sliced view would make every row > 0 read at the wrong
        # stride. vLLM asserts strict contiguity for the same reason.
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q.contiguous(),
            kv_cache=kv_cache,
            workspace_buffer=self._get_workspace(q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_table.contiguous(),
            seq_lens=cache_seqlens.contiguous(),
            max_seq_len=int(max_seq_len),
            bmm1_scale=softmax_scale,
            bmm2_scale=bmm2_scale,
            return_lse=False,
        )
        return out, None
