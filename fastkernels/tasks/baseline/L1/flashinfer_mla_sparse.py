"""FlashInfer sparse-MLA attention kernels (Blackwell sm100, FP8 KV cache).

vLLM's ``FLASHINFER_MLA_SPARSE`` backend, which its CUDA platform selects for a
DSA model (``index_topk`` present, ``qk_nope_head_dim in [128, 192]``) as soon as
the KV cache dtype is quantized -- ``_get_backend_priorities``
(vllm/platforms/cuda.py:98-116) puts ``FLASHINFER_MLA_SPARSE`` ahead of
``FLASHMLA_SPARSE`` for an fp8 cache. Verified on nvidia/GLM-5.2-NVFP4 with
``kv_cache_dtype=fp8_e4m3``:

    Using FLASHINFER_MLA_SPARSE attention backend out of potential backends:
    ['FLASHINFER_MLA_SPARSE'].
    Using HND KV cache layout for FLASHINFER_MLA_SPARSE backend.
    Using standard fp8 KV cache format. To use DeepSeek's fp8_ds_mla KV cache
    format, please set `--attention-backend FLASHMLA_SPARSE`
    Using TRTLLM_RAGGED MLA prefill backend.

Two kernels, one per half of ``MLAAttention.forward_impl``:

* :class:`FlashInferMLASparseDecode` — the sparse top-k MQA path
  (``FlashInferMLASparseImpl.forward_mqa``), used for decode tokens and, when
  the prefill is longer than ``index_topk``, prefill tokens too.
* :class:`TrtllmRaggedPrefill` — the dense MHA prefill path
  (``TrtllmRaggedPrefillBackend.run_prefill_new_tokens`` /
  ``run_prefill_context_chunk``), used when every prefill sequence is at most
  ``index_topk`` long, where sparse attention degenerates to dense.

Note this backend's KV layout is the **plain per-tensor fp8** one --
``[num_blocks, block_size, 576]`` ``float8_e4m3fn``, i.e. the same 576-dim slot
as BF16 but one byte per element -- NOT DeepSeek's 656-byte ``fp8_ds_mla``
block-scaled layout, which this backend explicitly rejects
(``supports_combination``: "FLASHINFER_MLA_SPARSE SM10 does not support
fp8_ds_mla kv-cache dtype").
"""

from __future__ import annotations

import torch
import torch.nn as nn

# vLLM's ``VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`` default. The buffer is shared
# by every layer (vLLM keeps one module-level ``_fi_sparse_workspace``), so
# fastkernels does the same rather than paying it per layer.
_WORKSPACE_BYTES = 512 * 1024 * 1024

_sparse_workspace: torch.Tensor | None = None
_ragged_workspace: torch.Tensor | None = None


def flashinfer_mla_sparse_available() -> bool:
    """True iff vLLM's FLASHINFER_MLA_SPARSE backend would be usable here.

    Matches ``FlashInferMLASparseTRTLLMBackend.supports_compute_capability``
    (sm100 exactly) plus the presence of both flashinfer entry points. The
    ``index_topk`` / ``qk_nope_head_dim`` parts of ``supports_combination`` are
    model properties and are checked by the caller.
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from flashinfer.decode import (  # noqa: F401
            trtllm_batch_decode_with_kv_cache_mla,
        )
        from flashinfer.prefill import trtllm_ragged_attention_deepseek  # noqa: F401

        return True
    except ImportError:
        return False


def _sparse_ws(device: torch.device) -> torch.Tensor:
    """Shared int8 workspace for the sparse MLA decode kernel.

    int8 (not uint8) because FlashInfer's CuteDSL MLA-decode tactic requires a
    signed workspace while the trtllm-gen path views it as uint8 — the same
    reasoning as vLLM's ``_get_workspace_buffer``.
    """
    global _sparse_workspace
    if _sparse_workspace is None or _sparse_workspace.device != device:
        _sparse_workspace = torch.zeros(
            _WORKSPACE_BYTES, dtype=torch.int8, device=device,
        )
    return _sparse_workspace


def _ragged_ws(device: torch.device) -> torch.Tensor:
    """Shared uint8 workspace for the ragged prefill kernel.

    vLLM allocates this one as uint8 through its workspace manager
    (``TrtllmRaggedPrefillBackend.__init__``), and keeps it separate from the
    decode workspace, so the two kernels never alias.
    """
    global _ragged_workspace
    if _ragged_workspace is None or _ragged_workspace.device != device:
        _ragged_workspace = torch.zeros(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=device,
        )
    return _ragged_workspace


class FlashInferMLASparseDecode(nn.Module):
    """Sparse top-k MQA over an fp8 paged MLA cache (trtllm-gen).

    Mirrors ``FlashInferMLASparseImpl.forward_mqa``
    (vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:388-470): the
    per-token top-k *global slot* indices are handed to the kernel as a
    ``block_tables`` of page size 1, so ``seq_lens`` is the per-token count of
    valid slots and ``max_seq_len`` is ``index_topk``.

    Every query token carries its own top-k row, so the ``q_len_per_request``
    dim is a bare ``unsqueeze(1)`` — vLLM does the same and notes the
    multi-token grouping is a perf-only layout deferred until MTP is validated.
    """

    def __init__(self, qk_nope_head_dim: int, qk_rope_head_dim: int,
                 kv_lora_rank: int):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank

    def forward(
        self,
        q: torch.Tensor,              # [N, H, 576] fp8_e4m3 (ql_nope || q_pe)
        kv_cache: torch.Tensor,       # [num_blocks, block_size, 576] fp8_e4m3
        topk_slots: torch.Tensor,     # [N, topk] int32 global slots (-1 = pad)
        seq_lens: torch.Tensor,       # [N] int32 valid slots per token
        topk_tokens: int,
        bmm1_scale: float,
        bmm2_scale: float,
    ) -> torch.Tensor:
        """Returns ``[N, H, kv_lora_rank]`` bf16 (pre ``v_up_proj``)."""
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=_sparse_ws(q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_slots.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=topk_tokens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sparse_mla_top_k=topk_tokens,
            return_lse=False,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.view(-1, out.shape[-2], out.shape[-1])


class TrtllmRaggedPrefill(nn.Module):
    """Dense varlen MHA prefill for MLA (trtllm-gen ragged, DeepSeek dims).

    Mirrors ``TrtllmRaggedPrefillBackend``: ``bmm1_scale=self.scale``,
    ``bmm2_scale=1.0``, ``o_sf_scale=1.0``, ``window_left=-1``,
    ``enable_pdl=False``, and the LSE transposed from ``(q_len, num_heads)`` to
    ``(num_heads, q_len)`` for the merge step. This is the backend vLLM's
    ``get_mla_prefill_backend`` selects on sm100 for GLM-5.2's
    ``(qk_nope=192, rope=64, v=256)`` dims — NOT FlashAttention, whose different
    accumulation order would show up in every prefill token.
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,               # [T, H, 256] bf16
        k: torch.Tensor,               # [T, H, 256] bf16
        v: torch.Tensor,               # [T, H, 256] bf16
        seq_lens: torch.Tensor,        # [B] int32 per-request KV length
        cu_seq_lens_q: torch.Tensor,   # [B+1] int32
        cu_seq_lens_kv: torch.Tensor,  # [B+1] int32
        max_q_len: int,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        out = torch.empty(
            q.shape[0], q.shape[1], v.shape[2],
            device=q.device, dtype=q.dtype,
        )
        ret = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=_ragged_ws(q.device),
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=seq_lens.shape[0],
            window_left=-1,
            cum_seq_lens_q=cu_seq_lens_q,
            cum_seq_lens_kv=cu_seq_lens_kv,
            enable_pdl=False,
            is_causal=is_causal,
            return_lse=return_lse,
            out=out,
        )
        if isinstance(ret, tuple):
            # (q_len, num_heads) -> (num_heads, q_len), matching vLLM so the
            # LSE feeds ``merge_attn_states`` in the layout it expects.
            return ret[0], ret[1].transpose(0, 1).contiguous()
        return ret


class QuantFp8MLAQuery(nn.Module):
    """Per-tensor fp8-quantize the absorbed 576-D MLA query.

    The counterpart of vLLM's ``_DecodeConcatQuantFP8``
    (mla_attention.py:1258-1289), applied by ``forward_impl`` when the KV cache
    is quantized and the impl sets ``supports_quant_query_input`` — the
    trtllm-gen sparse MLA kernel requires q and the cache to share a dtype
    (mixed bf16+fp8 is unsupported). vLLM's variant exists to *fuse* the
    ``cat(ql_nope, q_pe)`` with the quantization; fastkernels' caller already
    holds the concatenated query (``_absorb_q_to_latent``), so it quantizes that
    directly — same arithmetic, one copy fewer.

    Static per-tensor scale: ``QuantFP8.forward_cuda`` with a provided ``scale``
    dispatches to ``ops.scaled_fp8_quant`` ->
    ``torch.ops._C.static_scaled_fp8_quant``, so call that same CUDA kernel here
    rather than an equivalent-looking clamp/divide in PyTorch. ``scale`` is
    ``layer._q_scale``, which is 1.0 unless the checkpoint carries q/k
    calibration — nvidia/GLM-5.2-NVFP4 does not, and vLLM says so at load:
    "Checkpoint does not provide a q scaling factor ... Using KV cache scaling
    factor 1.0 for fp8_e4m3".
    """

    def forward(
        self,
        q: torch.Tensor,         # [N, H, kv_lora_rank + qk_rope_head_dim]
        scale: torch.Tensor,     # [1] fp32
    ) -> torch.Tensor:
        from vllm import _custom_ops as ops

        flat = q.reshape(q.shape[0], -1)
        out, _ = ops.scaled_fp8_quant(flat, scale)
        return out.view(q.shape)
