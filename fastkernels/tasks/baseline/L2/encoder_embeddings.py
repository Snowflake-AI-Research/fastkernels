"""Shared token/position/type embedding wiring for encoder models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L1.layer_norm import LayerNorm

TOKEN_TYPE_SHIFT = 30


def encode_token_type_ids(input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> None:
    input_ids[: token_type_ids.shape[0]].bitwise_or_(token_type_ids << TOKEN_TYPE_SHIFT)


def decode_token_type_ids(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids_mask = (
        torch.ones_like(input_ids, dtype=torch.int32, device=input_ids.device)
        << TOKEN_TYPE_SHIFT
    )
    tokens_mask = ids_mask.bitwise_not()
    token_type_ids = input_ids.bitwise_and(ids_mask) >> TOKEN_TYPE_SHIFT
    return input_ids.bitwise_and(tokens_mask), token_type_ids


def create_roberta_position_ids_from_input_ids(
    input_ids: torch.Tensor,
    padding_idx: int,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    mask = input_ids.ne(padding_idx).int()
    incremental = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental.long() + padding_idx


class EncoderEmbeddingsBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        word_padding_idx = self._word_embedding_padding_idx(config)
        position_padding_idx = self._position_embedding_padding_idx(config)

        self.word_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=word_padding_idx,
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=position_padding_idx,
        )
        self.token_type_embeddings = Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.LayerNorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def _word_embedding_padding_idx(self, config) -> int | None:
        return None

    def _position_embedding_padding_idx(self, config) -> int | None:
        return None

    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        past_key_values_length: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids, token_type_ids = decode_token_type_ids(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = (
            inputs_embeds
            + self.token_type_embeddings(token_type_ids.to(device=input_ids.device))
        )
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids.to(device=input_ids.device))
        return self.LayerNorm(embeddings)

    def forward_with_token_type_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, : input_ids.size(1)].expand(input_ids.size())
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids.to(input_ids.device))
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids.to(input_ids.device))
        return self.LayerNorm(embeddings)


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
