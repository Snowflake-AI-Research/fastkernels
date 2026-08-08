"""Neutralize empty-context prefill chunks before ``merge_attn_states``.

Vendored verbatim from vLLM's ``vllm.v1.attention.ops.triton_merge_attn_states``
(only change: import stock ``triton`` instead of ``vllm.triton_utils``, and drop
the ``float8_info`` global which is only used by the merge kernel, not here).
"""

import torch
import triton
import triton.language as tl


def mask_empty_context(
    lse: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    context_start_loc: torch.Tensor,
) -> None:
    """Neutralize context chunks that cover no keys before merging.

    A prefill query whose context chunk is empty attended to no keys, so its
    partial attention is undefined: the backend leaves the output rows as
    uninitialized scratch (which may hold NaN/Inf) even when it reports an LSE
    of -inf. Sanitize both here so ``merge_attn_states`` can stay generic:
    force the LSE to -inf (zero softmax weight) and zero the undefined output
    rows (so a zero weight cannot combine with NaN/Inf). Emptiness is derived
    from the context offsets, not from the -inf LSE, so no merge kernel has to
    reason about undefined partials.

    Args:
        lse: Chunk log-sum-exp, shape [num_heads, num_tokens].
        output: Chunk attention output, shape [num_tokens, num_heads, ...].
        query_start_loc: Prefill query cumulative offsets, shape [num_reqs + 1].
        context_start_loc: Chunk context cumulative offsets,
            shape [num_reqs + 1]; an empty chunk has a zero-length span.
    """
    num_heads, num_tokens = lse.shape
    num_reqs = query_start_loc.shape[0] - 1
    block_size = 128
    # Reserve the worst-case number of request-local blocks.
    num_query_blocks = num_tokens // block_size + num_reqs
    is_empty = torch.zeros(num_tokens, dtype=torch.bool, device=lse.device)
    mask_empty_context_kernel[(num_query_blocks,)](
        lse,
        is_empty,
        query_start_loc,
        context_start_loc,
        lse.stride(0),
        lse.stride(1),
        num_reqs,
        NUM_HEADS=num_heads,
        BLOCK_SIZE=block_size,
        BLOCK_HEADS=8,
        num_warps=8,
    )
    output.masked_fill_(is_empty[:, None, None], 0.0)


@triton.jit
def mask_empty_context_kernel(
    lse,
    is_empty,
    query_start_loc,
    context_start_loc,
    lse_head_stride,
    lse_token_stride,
    num_reqs,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    query_block_idx = tl.program_id(0)

    lanes = tl.arange(0, 32)
    chunk_start = 0
    req_idx = 0
    req_idx_found = False
    while (chunk_start < num_reqs) & (not req_idx_found):
        req_offsets = chunk_start + lanes
        req_mask = req_offsets < num_reqs
        query_starts = tl.load(query_start_loc + req_offsets, mask=req_mask)
        # Assume the worst-case number of blocks for each request.
        req_block_starts = query_starts // BLOCK_SIZE + req_offsets
        matched_idx = tl.sum(
            (req_mask & (req_block_starts <= query_block_idx)).to(tl.int32)
        )
        # matched_idx == 32 means the match is past this warp chunk.
        req_idx = chunk_start + matched_idx - 1
        req_idx_found = matched_idx < 32
        chunk_start += 32

    query_start = tl.load(query_start_loc + req_idx)
    query_end = tl.load(query_start_loc + req_idx + 1)
    query_len = query_end - query_start
    req_first_block = query_start // BLOCK_SIZE + req_idx
    block_in_req = query_block_idx - req_first_block
    token_offset = block_in_req * BLOCK_SIZE
    if token_offset >= query_len:
        return

    context_start = tl.load(context_start_loc + req_idx)
    context_end = tl.load(context_start_loc + req_idx + 1)
    if context_start != context_end:
        return

    token_offsets = token_offset + tl.arange(0, BLOCK_SIZE)
    token_indices = query_start + token_offsets
    token_lse_offsets = token_indices * lse_token_stride
    valid_tokens = token_offsets < query_len
    tl.store(is_empty + token_indices, True, mask=valid_tokens)
    head_offsets = tl.arange(0, BLOCK_HEADS)
    for head_start in range(0, NUM_HEADS, BLOCK_HEADS):
        head_indices = head_start + head_offsets
        lse_ptrs = (
            lse + head_indices[:, None] * lse_head_stride + token_lse_offsets[None, :]
        )
        valid_heads = head_indices < NUM_HEADS
        tl.store(
            lse_ptrs,
            float("-inf"),
            mask=valid_heads[:, None] & valid_tokens[None, :],
        )
