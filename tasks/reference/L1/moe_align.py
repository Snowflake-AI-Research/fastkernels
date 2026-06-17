"""Semantic PyTorch reference for MoE token-to-expert alignment."""

from __future__ import annotations

import torch
import torch.nn as nn

class MoeAlign(nn.Module):
    """Token-to-expert alignment with per-expert block padding."""

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

        flat = topk_ids.view(-1).to(torch.int32)
        numel = flat.numel()
        sorted_chunks: list[torch.Tensor] = []
        block_experts: list[int] = []

        for expert in range(num_experts):
            token_ids = torch.nonzero(flat == expert, as_tuple=False).flatten().to(torch.int32)
            if token_ids.numel() == 0:
                continue
            pad = (-token_ids.numel()) % block_size
            if pad:
                padding = torch.full(
                    (pad,), numel, dtype=torch.int32, device=topk_ids.device,
                )
                token_ids = torch.cat([token_ids, padding], dim=0)
            sorted_chunks.append(token_ids)
            block_experts.extend([expert] * (token_ids.numel() // block_size))

        if sorted_chunks:
            sorted_token_ids = torch.cat(sorted_chunks, dim=0)
        else:
            sorted_token_ids = torch.empty(0, dtype=torch.int32, device=topk_ids.device)
        expert_ids = torch.tensor(block_experts, dtype=torch.int32, device=topk_ids.device)
        num_tokens_post_padded = torch.tensor(
            [sorted_token_ids.numel()], dtype=torch.int32, device=topk_ids.device,
        )
        return sorted_token_ids, expert_ids, num_tokens_post_padded
