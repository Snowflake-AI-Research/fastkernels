"""BERT embedding stack."""

from __future__ import annotations

import torch

from .encoder_embeddings import EncoderEmbeddingsBase


class BertEmbeddings(EncoderEmbeddingsBase):
    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            seq_len = input_ids.size(1)
            return self.position_ids[
                :,
                past_key_values_length: past_key_values_length + seq_len,
            ]

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided when input_ids is None")
        input_shape = inputs_embeds.size()[:-1]
        seq_len = input_shape[1]
        return self.position_ids[
            :,
            past_key_values_length: past_key_values_length + seq_len,
        ]
