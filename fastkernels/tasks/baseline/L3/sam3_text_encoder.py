"""Text encoder for SAM3.

Tokenizes text captions, runs them through a causal text transformer, and
resizes the output to match the detection model's d_model dimension.

Reference: sam3/model/text_encoder_ve.py VETextEncoder + TextTransformer
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..L1.embedding import Embedding
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .sam3_text_encoder_layer import Sam3TextEncoderLayer


class Sam3TextEncoder(nn.Module):
    """Text encoder for SAM3.

    Tokenizes input strings, runs a causal transformer, and projects hidden
    states to the detection model's dimension.

    Args:
        d_model: Detection model dimension (output dimension).
        width: Transformer hidden dimension.
        heads: Number of attention heads.
        layers: Number of transformer layers.
        context_length: Maximum token sequence length.
        vocab_size: Vocabulary size.
    """

    def __init__(
        self,
        d_model: int = 256,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
    ):
        super().__init__()
        self.context_length = context_length
        self.width = width

        self.token_embedding = Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))

        self.transformer_layers = nn.ModuleList([
            Sam3TextEncoderLayer(width, heads) for _ in range(layers)
        ])
        self.ln_final = LayerNorm(width)

        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        self.register_buffer("attn_mask", mask, persistent=False)

        self.resizer = Linear(width, d_model, bias=True)

    def forward(
        self,
        tokenized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode tokenized text.

        Args:
            tokenized: (B, seq_len) integer token IDs.

        Returns:
            (text_attention_mask, text_memory_resized, inputs_embeds):
                text_attention_mask: (B, seq_len) boolean, True = padding.
                text_memory_resized: (seq_len, B, d_model) seq-first features.
                inputs_embeds: (seq_len, B, width) raw embeddings, seq-first.
        """
        seq_len = tokenized.shape[1]
        text_attention_mask = (tokenized == 0)  # True where padding

        inputs_embeds = self.token_embedding(tokenized)  # (B, seq, width)
        x = inputs_embeds + self.positional_embedding[:seq_len]

        attn_mask = self.attn_mask[:seq_len, :seq_len] if self.attn_mask is not None else None

        for layer in self.transformer_layers:
            x = layer(x, attn_mask=attn_mask)

        text_memory = self.ln_final(x)  # (B, seq, width)

        text_memory_sf = text_memory.transpose(0, 1)  # (seq, B, width)
        text_memory_resized = self.resizer(text_memory_sf)  # (seq, B, d_model)

        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)
