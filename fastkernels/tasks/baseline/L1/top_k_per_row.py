"""Top-K per row selection for DSA indexer using vLLM's CUDA kernels."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import vllm._custom_ops  # noqa: F401 — registers torch.ops._C
import flashinfer as _flashinfer
from flashinfer import TopKTieBreak as _TopKTieBreak


# vLLM's DSA decode indexer uses a radix-histogram top-k fast path for the
# common index_topk sizes; see ``vllm/model_executor/layers/sparse_attn_indexer.py:337-385``.
_RADIX_TOPK_SIZES = (512, 1024, 2048)
_RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024  # bytes (matches vLLM RADIX_TOPK_WORKSPACE_SIZE)


# --- DETERMINISTIC DSA top-k (diagnostic) --------------------------------
# vLLM's radix top-k kernels (cooperative_topk / persistent_topk /
# top_k_per_row_*) emit the correct top-k SET but in a NONDETERMINISTIC ORDER
# (atomic-append), plus a rarer SET race at exact score ties on the k-th
# boundary. Downstream ``flash_mla_sparse_fwd`` is order-sensitive, so this
# makes greedy decode run-to-run nondeterministic at seq>index_topk — for vLLM
# too (proven by injection). ``flashinfer.top_k`` is a radix top-k with a
# ``deterministic`` flag AND a ``tie_break`` knob, giving a bit-reproducible
# SET *and* ORDER while staying faster than ``torch.topk`` (esp. at long ctx).
# Enable with ``FASTKERNELS_DSA_DETERMINISTIC_TOPK=1`` (default off = match
# vLLM's native nondeterministic kernels).
#
# DIAGNOSTIC ONLY, and incompatible with decode CUDA graphs: the flashinfer
# top-k path aborts during graph replay, so use it together with
# ``--enforce-eager`` (which is how ``forced_decode.py`` drives it).
_DETERMINISTIC_TOPK = os.environ.get("FASTKERNELS_DSA_DETERMINISTIC_TOPK", "0") != "0"


def _deterministic_topk_indices(
    logits: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Deterministic per-row top-k indices within the window ``[ks, ke)``.

    Masks out-of-window positions (this also neutralizes the
    ``clean_logits=False`` garbage region) to a finite sentinel, then uses
    ``flashinfer.top_k(sorted=True, deterministic=True, tie_break=SMALL)`` for a
    bit-reproducible set+order. Rows with fewer than ``topk`` valid positions
    get ``-1`` padding (selected sentinels are mapped back to ``-1``).
    Returns ``[R, topk]`` int32.
    """
    R, N = logits.shape
    dev = logits.device
    cols = torch.arange(N, device=dev)
    valid = (cols.unsqueeze(0) >= ks.unsqueeze(1).to(torch.long)) & (
        cols.unsqueeze(0) < ke.unsqueeze(1).to(torch.long)
    )
    sentinel = torch.finfo(logits.dtype).min
    lm = torch.where(valid, logits, torch.full_like(logits, sentinel))
    vals, idx = _flashinfer.top_k(
        lm, topk, sorted=False, deterministic=True,
        tie_break=int(_TopKTieBreak.SMALL),
    )
    idx = idx.to(torch.int32)
    # Under-k rows selected the sentinel -> mark invalid.
    idx = torch.where(vals <= sentinel, torch.full_like(idx, -1), idx)
    # Return indices RELATIVE to the per-row window start ``ks`` — this is what
    # the native ``top_k_per_row_prefill`` kernel emits (verified: window
    # [2000,5000) -> [0,3000), not absolute columns). ``convert_indices`` then
    # computes ``block_id = idx // block_size`` against the PER-REQUEST block
    # table, so an ABSOLUTE column of a sequence at a non-zero packed offset
    # (e.g. the 2nd+ sequence in a batch) overflows that request's block table
    # and gets dropped to -1 — the packed-batch DSA bug. ``flashinfer.top_k``
    # returns absolute columns of ``lm``, so subtract ``ks`` here. Decode passes
    # ``ks=0`` so this is a no-op there.
    ks_row = ks.reshape(-1, 1).to(idx.dtype)
    idx = torch.where(idx >= 0, idx - ks_row, idx)
    # Emit in INDEX-ASCENDING order (padding -1 last). This is value-INSENSITIVE:
    # for a given selected SET the order is identical regardless of tiny
    # cross-engine logit differences (batched GEMM tiling). A value-sorted order
    # would flip under those ULP diffs and desync the order-sensitive sparse
    # attention across engines / batch compositions.
    _INTMAX = 2147483647
    t = torch.where(idx >= 0, idx, torch.full_like(idx, _INTMAX))
    t, _ = torch.sort(t, dim=-1)
    return torch.where(t == _INTMAX, torch.full_like(t, -1), t)


class TopKPerRow(nn.Module):
    """Top-K per row selection for sparse attention indexer.

    Uses ``torch.ops._C.top_k_per_row_prefill`` and
    ``torch.ops._C.top_k_per_row_decode`` CUDA kernels for high performance.
    Decode additionally dispatches to the radix ``cooperative_topk`` /
    ``persistent_topk`` kernels for ``topk in {512,1024,2048}`` (the fast path
    vLLM takes for DeepSeek-V3.2 / GLM-5.2, whose ``index_topk`` is 2048).
    """

    def __init__(self):
        super().__init__()
        self._radix_ws: dict[torch.device, torch.Tensor] = {}

    def ensure_workspaces(self, device: torch.device) -> None:
        """Materialize the radix top-k workspace before CUDA graph capture.

        The radix kernels reduce into this buffer atomically. Allocated lazily
        on first use it lands in the capturing graph's private memory pool, and
        every subsequently captured batch size then replays an atomic into
        another graph's pool -- which faults at replay
        (``cudaErrorIllegalAddress`` / ``cudaErrorInvalidAddressSpace``). The
        engine calls this after KV cache allocation, before capture.
        """
        self._radix_workspace(device)

    def _radix_workspace(self, device: torch.device) -> torch.Tensor:
        ws = self._radix_ws.get(device)
        if ws is None:
            ws = torch.empty(_RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8,
                             device=device)
            self._radix_ws[device] = ws
        return ws

    def forward_prefill(
        self,
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Top-K within variable row boundaries.

        Args:
            logits: ``[M, max_seq_len]`` float32 logits.
            cu_seqlen_ks: ``[M]`` row start offsets (inclusive).
            cu_seqlen_ke: ``[M]`` row end offsets (exclusive).
            topk: number of indices to keep per row.
            out: optional ``[M, topk]`` int32 buffer to write into (e.g. a
                slice of the shared ``topk_indices_buffer``). Must be a
                contiguous full-width slice. Avoids a fresh alloc + copy on
                the indexer hot path; matches vLLM writing straight into
                ``topk_indices_buffer``.

        Returns:
            ``indices``: ``[M, topk]`` int32 (``out`` if provided).
        """
        M = logits.shape[0]
        if out is not None:
            indices = out
            indices.fill_(-1)
        else:
            indices = torch.full(
                (M, topk), -1, dtype=torch.int32, device=logits.device,
            )
        if M == 0:
            return indices

        if _DETERMINISTIC_TOPK:
            det = _deterministic_topk_indices(logits, cu_seqlen_ks, cu_seqlen_ke, topk)
            indices.copy_(det)
            return indices

        torch.ops._C.top_k_per_row_prefill(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            indices,
            M,
            logits.stride(0),
            logits.stride(1),
            topk,
        )
        return indices

    def forward_decode(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        next_n: int,
        topk: int,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Top-K for decode rows with per-sequence length caps.

        Args:
            logits: ``[B * next_n, max_seq_len]`` float32 logits.
            seq_lens: ``[B]`` current sequence lengths.
            next_n: speculative tokens per sequence (row stride).
            topk: number of indices per row.
            max_seq_len: max KV length (sizes internal radix work; falls back to
                the logits column dim).

        Returns:
            ``indices``: ``[B * next_n, topk]`` int32.
        """
        total_rows = logits.shape[0]
        indices = torch.full(
            (total_rows, topk), -1, dtype=torch.int32, device=logits.device,
        )
        if total_rows == 0:
            return indices

        if _DETERMINISTIC_TOPK and logits.is_cuda:
            # Per-row causal length (same cap as the radix path: seq_len for
            # next_n==1, else max(0, seq_len - next_n + 1 + j)). Window is
            # [0, row_len); flashinfer gives a deterministic set+order.
            if next_n == 1:
                row_lens = seq_lens.to(torch.int32)
            else:
                B = seq_lens.numel()
                j = torch.arange(next_n, device=seq_lens.device, dtype=torch.int32)
                row_lens = (
                    seq_lens.to(torch.int32).view(B, 1) - next_n + 1 + j.view(1, next_n)
                ).clamp_min_(0).reshape(-1)
            ks = torch.zeros(total_rows, dtype=torch.int32, device=logits.device)
            indices.copy_(_deterministic_topk_indices(logits, ks, row_lens, topk))
            return indices

        # Radix fast path — matches vLLM ``sparse_attn_indexer.py:337-385``. The
        # radix kernels take per-ROW lengths (not next_n), so expand the per-seq
        # ``seq_lens`` to per-row caps identical to the generic kernel's cap
        # (sampler.cu:591: ``max(0, seq_len - next_n + j + 1)``). This is the
        # branch vLLM actually executes for GLM-5.2 (index_topk=2048).
        if logits.is_cuda and topk in _RADIX_TOPK_SIZES:
            if next_n == 1:
                row_lens = seq_lens.to(torch.int32).contiguous()
            else:
                B = seq_lens.numel()
                j = torch.arange(next_n, device=seq_lens.device, dtype=torch.int32)
                row_lens = (
                    seq_lens.to(torch.int32).view(B, 1) - next_n + 1 + j.view(1, next_n)
                ).clamp_min_(0).reshape(-1).contiguous()
            # vLLM passes DIFFERENT last arguments to the two radix kernels:
            # ``cooperative_topk`` gets ``attn_metadata.max_seq_len`` (the batch's
            # max context) and ``persistent_topk`` gets ``logits.shape[1]`` (the
            # fixed buffer width). Passing one value to both changes the radix
            # range on whichever kernel is chosen.
            coop_msl = (
                int(max_seq_len) if max_seq_len is not None else logits.shape[1]
            )
            ws = self._radix_workspace(logits.device)
            cap_major = torch.cuda.get_device_capability(logits.device)[0]
            use_cooperative = (
                total_rows <= 32
                and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
                and cap_major >= 9
            )
            if use_cooperative:
                torch.ops._C.cooperative_topk(
                    logits, row_lens, indices, ws, topk, coop_msl)
            else:
                torch.ops._C.persistent_topk(
                    logits, row_lens, indices, ws, topk, logits.shape[1])
            return indices

        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            indices,
            total_rows,
            logits.stride(0),
            logits.stride(1),
            topk,
        )
        return indices
