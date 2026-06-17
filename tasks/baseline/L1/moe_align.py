"""MoE token-to-expert alignment with block padding.

Uses a custom CUDA kernel for high-performance, CUDA-graph-compatible
token-to-expert alignment. Supports a naive fast path that skips the full sort
when the number of tokens is very small relative to the number of experts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton

from .csrc import _C


class MoeAlign(nn.Module):
    """MoE token-to-expert alignment using sgl_kernel.

    Pre-allocates output buffers for reuse and CUDA graph compatibility.
    """

    def __init__(self):
        super().__init__()
        self._sorted_token_ids = None
        self._expert_ids = None
        self._num_tokens_post_padded = None
        self._cumsum_buffer = None
        self._naive_num_tokens_post_padded = None

    def _ensure_buffers(self, max_padded, max_blocks, num_experts, device):
        if (self._sorted_token_ids is None
                or self._sorted_token_ids.size(0) < max_padded):
            self._sorted_token_ids = torch.empty(
                max_padded, dtype=torch.int32, device=device,
            )
        if (self._expert_ids is None
                or self._expert_ids.size(0) < max_blocks):
            self._expert_ids = torch.empty(
                max_blocks, dtype=torch.int32, device=device,
            )
        if (self._num_tokens_post_padded is None
                or self._num_tokens_post_padded.device != device):
            self._num_tokens_post_padded = torch.zeros(
                1, dtype=torch.int32, device=device,
            )
        if (self._cumsum_buffer is None
                or self._cumsum_buffer.size(0) < num_experts + 1):
            self._cumsum_buffer = torch.zeros(
                num_experts + 1, dtype=torch.int32, device=device,
            )

    def _naive_forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Fast path: skip full alignment when tokens * top_k is very small."""
        numel = topk_ids.numel()
        max_num_tokens_padded = numel * block_size
        expert_ids = topk_ids.view(-1).to(torch.int32)
        if (self._naive_num_tokens_post_padded is None
                or self._naive_num_tokens_post_padded.device != topk_ids.device):
            self._naive_num_tokens_post_padded = torch.empty(
                1, dtype=torch.int32, device=topk_ids.device,
            )
        self._naive_num_tokens_post_padded.fill_(max_num_tokens_padded)
        return None, expert_ids, self._naive_num_tokens_post_padded

    def forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        naive: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if naive:
            return self._naive_forward(topk_ids, block_size)

        numel = topk_ids.numel()
        if numel < num_experts:
            max_padded = numel * block_size
        else:
            max_padded = numel + num_experts * (block_size - 1)
        max_blocks = triton.cdiv(max_padded, block_size)

        self._ensure_buffers(max_padded, max_blocks, num_experts, topk_ids.device)

        sorted_token_ids = self._sorted_token_ids[:max_padded]
        expert_ids = self._expert_ids[:max_blocks]

        _C.moe_align_block_size(
            topk_ids.view(-1).contiguous(),
            num_experts, block_size,
            sorted_token_ids, expert_ids,
            self._num_tokens_post_padded,
            self._cumsum_buffer[:num_experts + 1],
            True,
        )

        return sorted_token_ids, expert_ids, self._num_tokens_post_padded
