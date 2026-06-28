"""BERT encoder layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.encoder_attention import EncoderAttention
from ..L2.encoder_mlp import EncoderIntermediate, EncoderOutput


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = EncoderAttention(config)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention.forward_with_attention_mask(
            hidden_states,
            attention_mask=attention_mask,
        )
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        attention_output = self.attention.forward_varlen(hidden_states, cu_seqlens, max_seqlen)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)
