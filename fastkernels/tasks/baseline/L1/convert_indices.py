"""Convert request-local token indices to global paged cache slots.

Supports both decode (block_table lookup) and prefill workspace
(direct offset mapping) modes, matching vllm's
``_convert_req_index_to_global_index_kernel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    valid_count_ptr,
    prefill_request_id_ptr,
    workspace_starts_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)

    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

    # Per-row count of valid (non ``-1``) slots, accumulated across column tiles
    # with one atomic per tile. Mirrors vLLM's ``COUNT_VALID`` branch
    # (v1/attention/backends/mla/sparse_utils.py:114-117); the FlashInfer sparse
    # MLA decode kernel takes these counts as its ``seq_lens``, so they must be
    # computed in the same pass rather than from a separate reduction over
    # ``out`` (which would be a second kernel on the attention hot path).
    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


class ConvertIndicesToGlobal(nn.Module):
    """Map per-request token indices to global linear cache slots.

    Supports both simple decode-only mode (block_table lookup) and
    mixed prefill+decode mode with workspace offset mapping.
    """

    def forward(
        self,
        indices: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        req_ids: torch.Tensor | None = None,
        prefill_request_ids: torch.Tensor | None = None,
        prefill_workspace_starts: torch.Tensor | None = None,
        return_valid_counts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert local indices to global slot indices.

        Args:
            indices: ``[num_tokens, topk]`` int32.
            block_table: ``[num_reqs, max_blocks]`` int32.
            block_size: tokens per block.
            req_ids: ``[num_tokens]`` int32 — which request each token belongs to.
                If None, assumes identity mapping (token i -> request i).
            prefill_request_ids: ``[num_tokens]`` int32 — -1 for decode,
                >=0 for prefill (index into prefill_workspace_starts).
            prefill_workspace_starts: ``[num_prefills]`` int32 — workspace
                start offset per prefill request.
            return_valid_counts: also return ``[num_tokens]`` int32 counts of
                valid (non ``-1``) slots per row, computed in the same kernel
                pass. Needed by the FlashInfer sparse MLA decode kernel.

        Returns:
            ``global_indices``: ``[num_tokens, topk]`` int32, or
            ``(global_indices, valid_counts)`` when ``return_valid_counts``.
        """
        num_tokens, topk = indices.shape
        has_prefill = prefill_request_ids is not None and prefill_workspace_starts is not None

        if req_ids is None:
            req_ids = torch.arange(num_tokens, dtype=torch.int32, device=indices.device)

        BLOCK_N = min(128, topk)
        assert topk % BLOCK_N == 0

        max_num_blocks_per_req = block_table.shape[1]
        tiles_per_row = topk // BLOCK_N

        # Materialize contiguous copies BEFORE reading strides. The kernel
        # indexes with the strides passed here, so they must belong to the
        # exact tensors the kernel receives. ``block_table`` in eager decode is
        # a non-contiguous slice (``_eager_block_tables[:n, :bt_cols]`` whose
        # row stride is the full buffer width, not ``bt_cols``); taking strides
        # from the slice while passing a fresh ``.contiguous()`` copy made the
        # kernel read request rows at the wrong offset -> garbage global slots
        # for every request but the first (req 0 starts at 0 either way, so
        # single-sequence / offset-0 batches masked the bug).
        req_ids = req_ids.contiguous()
        block_table = block_table.contiguous()
        indices = indices.contiguous()

        out = torch.empty_like(indices)
        # Must be zero-initialized: the kernel accumulates into it atomically.
        valid_counts = (
            torch.zeros(num_tokens, dtype=torch.int32, device=indices.device)
            if return_valid_counts else None
        )

        bt_stride0, bt_stride1 = block_table.stride()
        ti_stride0, ti_stride1 = indices.stride()
        out_stride0, out_stride1 = out.stride()

        grid = (num_tokens, tiles_per_row)
        _convert_req_index_to_global_index_kernel[grid](
            req_ids,
            block_table,
            indices,
            out,
            valid_counts,
            prefill_request_ids if has_prefill else prefill_request_ids,
            prefill_workspace_starts if has_prefill else prefill_workspace_starts,
            max_num_blocks_per_req,
            block_size,
            BLOCK_N,
            has_prefill,
            return_valid_counts,
            bt_stride0, bt_stride1,
            ti_stride0, ti_stride1,
            out_stride0, out_stride1,
        )
        if return_valid_counts:
            return out, valid_counts
        return out
