"""CLIP MLP and text embeddings (L2).

CLIPMLP: Linear -> QuickGELU -> Linear (no TP, frozen encoder).
CLIPTextEmbeddings: token + position embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPTextConfig

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.quickgelu import QuickGELU


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)
