"""Attention blocks for encoder models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.flash_attn_varlen import FlashAttnVarlen
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


class EncoderSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.num_attention_heads}",
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Fused QKV, matching vLLM's BertSelfAttention, which builds a single
        # QKVParallelLinear and splits its output. Three separate projections
        # issue 3x the GEMM dispatches for identical math, and this forward is
        # host-dispatch-bound at small token counts. The checkpoint's separate
        # query/key/value tensors are concatenated at load time by
        # ``embedder_loader._fuse_qkv_keys``.
        self.qkv = Linear(config.hidden_size, 3 * self.all_head_size, bias=True)
        self.attn = DenseAttention(backend="sdpa")
        self.varlen_attn = FlashAttnVarlen()

    def _project_qkv(self, hidden_states: torch.Tensor):
        qkv = self.qkv(hidden_states)
        return qkv.split(
            [self.all_head_size, self.all_head_size, self.all_head_size],
            dim=-1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_attention_mask(hidden_states)

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = (
            t.view(
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            )
            for t in self._project_qkv(hidden_states)
        )
        context = self.attn(query, key, value, causal=False, attn_mask=attention_mask)
        return context.contiguous().view(batch_size, seq_len, self.all_head_size)

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        query, key, value = (
            t.view(
                hidden_states.size(0),
                self.num_attention_heads,
                self.attention_head_size,
            )
            for t in self._project_qkv(hidden_states)
        )
        context = self.varlen_attn(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.attention_head_size ** -0.5,
            causal=False,
        )
        if isinstance(context, tuple):
            context = context[0]
        return context.contiguous().view(hidden_states.size(0), self.all_head_size)


class EncoderSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, bias=True)
        # promote_fp32=False: vLLM's bert.py / roberta.py use a plain
        # nn.LayerNorm here (see encoder_embeddings for the full rationale).
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                   promote_fp32=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class EncoderAttention(nn.Module):
    self_attention_cls = EncoderSelfAttention
    self_output_cls = EncoderSelfOutput

    def __init__(self, config):
        super().__init__()
        self.self = self.self_attention_cls(config)
        self.output = self.self_output_cls(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.self(hidden_states)
        return self.output(attention_output, hidden_states)

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.self.forward_with_attention_mask(
            hidden_states,
            attention_mask=attention_mask,
        )
        return self.output(attention_output, hidden_states)

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        attention_output = self.self.forward_varlen(hidden_states, cu_seqlens, max_seqlen)
        return self.output(attention_output, hidden_states)
