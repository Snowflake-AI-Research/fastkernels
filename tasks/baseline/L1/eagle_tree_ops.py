"""EAGLE-3 tree-construction and tree-verify CUDA ops.

Vendored kernels (see csrc/eagle_utils.cu) ported from sglang's sgl-kernel
(Apache-2.0). Python wrappers mirror sglang's
sglang/srt/speculative/eagle_utils.py one-for-one but call into the kb_nano
JIT extension instead of `sgl_kernel`.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import List, Optional, Tuple

import torch

from .csrc import _C


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the (num_draft_token - 1) highest-scoring draft tokens across all
    expansion steps and return the parent table, sorted indices, and the
    actual draft token ids.
    """
    score_list_cat = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list_cat, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, device=parents_list[0].device, dtype=torch.long
        )

    return parent_list, top_scores_index, draft_tokens


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    """Build the verification tree and return tree mask, positions, and
    retrieve indices.

    Returns
    -------
    tree_mask, positions, retrive_index, retrive_next_token,
    retrive_next_sibling, draft_tokens
    """
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)

    bs = seq_lens.numel()
    device = seq_lens.device

    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            pass
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        # ``build_tree_efficient`` writes every [B, N, N] entry in this mode,
        # so avoid a separate fill kernel on the hot path.
        tree_mask = torch.empty(
            (num_verify_tokens * bs * num_verify_tokens,),
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            dtype=torch.bool,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf

    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    _C.build_tree_kernel_efficient(
        parent_list.to(dtype=torch.int64),
        top_scores_index,
        seq_lens,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        int(topk),
        int(spec_steps),
        int(num_verify_tokens),
        int(tree_mask_mode),
    )

    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def build_tree_kernel_efficient_with_metadata(
    verified_id: torch.Tensor,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping_draft: torch.Tensor,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    position_buf: Optional[torch.Tensor] = None,
    page_table_expand_buf: Optional[torch.Tensor] = None,
    cache_seqlens_expand_buf: Optional[torch.Tensor] = None,
):
    """Build the verification tree and FA3 expand metadata in one kernel.

    This is the hot-path variant for QLEN_ONLY tree attention. It avoids
    materializing the flat bool tree mask and the second metadata kernel.
    """
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    bs = seq_lens.numel()
    device = seq_lens.device

    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    positions = (
        position_buf
        if position_buf is not None
        else torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)
    )
    page_table_expand = (
        page_table_expand_buf
        if page_table_expand_buf is not None
        else torch.empty(
            (bs * num_verify_tokens, num_verify_tokens),
            device=device,
            dtype=torch.int32,
        )
    )
    cache_seqlens_expand = (
        cache_seqlens_expand_buf
        if cache_seqlens_expand_buf is not None
        else torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.int32)
    )

    _C.build_tree_kernel_efficient_with_metadata(
        parent_list.to(dtype=torch.int64),
        top_scores_index,
        seq_lens,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        slot_mapping_draft,
        page_table_expand,
        cache_seqlens_expand,
        int(topk),
        int(spec_steps),
        int(num_verify_tokens),
    )

    return (
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
        page_table_expand,
        cache_seqlens_expand,
    )


def verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Greedy verification of a draft tree against target predictions.

    Mutates `predicts`, `accept_index`, `accept_token_num`. See
    csrc/eagle_utils.cu :: VerifyTreeGreedy for shape semantics.
    """
    _C.verify_tree_greedy(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )


def build_tree_cascade_metadata(
    tree_mask: torch.Tensor,
    slot_mapping_draft: torch.Tensor,
    num_verify_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build FA3 expand metadata from the QLEN_ONLY tree mask.

    Returns ``(page_table_expand, cache_seqlens_expand)`` for the verify
    tree's draft-token attention pass. The page table keeps live ancestor
    slots first in ascending tree-column order, matching sglang's metadata.
    """
    num_rows = tree_mask.numel() // num_verify_tokens
    page_table_expand = torch.empty(
        num_rows, num_verify_tokens,
        device=tree_mask.device,
        dtype=torch.int32,
    )
    cache_seqlens_expand = torch.empty(
        num_rows, device=tree_mask.device, dtype=torch.int32,
    )
    _C.build_tree_cascade_metadata(
        tree_mask,
        slot_mapping_draft,
        page_table_expand,
        cache_seqlens_expand,
        int(num_verify_tokens),
    )
    return page_table_expand, cache_seqlens_expand
