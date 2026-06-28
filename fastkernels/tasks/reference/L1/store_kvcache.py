"""Semantic PyTorch reference for store_kvcache.

This file is used for specification/prompting and optional validation only.
It is not the production baseline and should not be used for reported speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StoreKVCache(nn.Module):
    """NHD layout store: [num_blocks, block_size, num_kv_heads, head_dim]."""

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        flat_k = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
        valid = slot_mapping >= 0
        slots = slot_mapping[valid].long()
        flat_k.index_copy_(0, slots, key[valid])
        flat_v.index_copy_(0, slots, value[valid])


class StoreKVCacheHND(nn.Module):
    """HND layout store: [num_blocks, num_kv_heads, block_size, head_dim]."""

    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        valid = slot_mapping >= 0
        slots = slot_mapping[valid].long()
        block_idx = slots // self.page_size
        slot_in_block = slots % self.page_size
        k_cache[block_idx, :, slot_in_block, :] = key[valid]
        v_cache[block_idx, :, slot_in_block, :] = value[valid]
