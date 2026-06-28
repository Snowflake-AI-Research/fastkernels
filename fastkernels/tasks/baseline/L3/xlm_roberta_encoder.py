"""XLM-RoBERTa encoder stack."""

from __future__ import annotations

import torch
import torch.nn as nn

from .xlm_roberta_layer import XLMRobertaLayer


class XLMRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([
            XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states

    def forward_with_attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module.forward_with_attention_mask(
                hidden_states,
                attention_mask=attention_mask,
            )
        return hidden_states

    def forward_varlen(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module.forward_varlen(hidden_states, cu_seqlens, max_seqlen)
        return hidden_states
