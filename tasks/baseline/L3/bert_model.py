"""BERT encoder-only model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L2.encoder_embeddings import BertEmbeddings
from .bert_encoder import BertEncoder


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )
        return self.encoder(embedding_output)

    def forward_with_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is None:
            seq_length = input_ids.size(1)
            positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
            positions = positions.expand(input_ids.size(0), seq_length)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        embedding_output = self.embeddings.forward_with_token_type_ids(
            input_ids=input_ids,
            position_ids=positions,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        extended_attention_mask = self._prepare_attention_mask(attention_mask, input_ids.device)
        return self.encoder.forward_with_attention_mask(
            embedding_output,
            attention_mask=extended_attention_mask,
        )

    def forward_varlen(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )
        return self.encoder.forward_varlen(embedding_output, cu_seqlens, max_seqlen)
