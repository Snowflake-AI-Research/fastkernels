"""T5 self-attention with TP-aware QKV projection and relative position bias (L2).

Mirrors vllm-omni's T5SelfAttention: QKVParallelLinear -> manual SDPA ->
RowParallelLinear, with T5-style relative position bias computed per-partition.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import T5Config

from ....infra.tp import _tp_size, _tp_rank
from ..L1.embedding import Embedding
from ..L1.linear import BMM
from ..L1.softmax import Softmax
from .parallel_linear import QKVParallelLinear, RowParallelLinear


class T5SelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = _tp_size()
        assert self.n_heads % tp_size == 0
        self.n_heads_per_partition = self.n_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )

        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large,
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        tp_rank = _tp_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        qkv = self.qkv_proj(hidden_states)
        q_size = self.n_heads_per_partition * self.d_kv
        kv_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split(
            [q_size, kv_size, kv_size], dim=-1,
        )

        query_states = query_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)

        scores = self.bmm(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=scores.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=scores.device, dtype=scores.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        scores += position_bias
        attn_weights = self.softmax(scores.float()).type_as(scores)
        attn_output = self.bmm(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        attn_output = self.o(attn_output)

        return attn_output, position_bias
