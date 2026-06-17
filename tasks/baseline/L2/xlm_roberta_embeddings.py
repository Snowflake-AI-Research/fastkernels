"""XLM-RoBERTa embedding stack."""

from __future__ import annotations

import torch

from .encoder_embeddings import (
    EncoderEmbeddingsBase,
    create_roberta_position_ids_from_input_ids,
)


class XLMRobertaEmbeddings(EncoderEmbeddingsBase):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id

    def _word_embedding_padding_idx(self, config) -> int | None:
        return config.pad_token_id

    def _position_embedding_padding_idx(self, config) -> int | None:
        return config.pad_token_id

    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            return create_roberta_position_ids_from_input_ids(
                input_ids,
                self.padding_idx,
                past_key_values_length,
            )

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided when input_ids is None")
        input_shape = inputs_embeds.size()[:-1]
        seq_len = input_shape[1]
        return torch.arange(
            self.padding_idx + 1,
            seq_len + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0).expand(input_shape)
