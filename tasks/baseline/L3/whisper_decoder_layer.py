"""Whisper decoder layer: self-attention + cross-attention + MLP.

Matches vLLM's WhisperDecoderLayer. Pre-norm residual connections:
  x = x + self_attn(layer_norm(x))
  x = x + cross_attn(layer_norm(x), encoder_hidden_states)
  x = x + mlp(layer_norm(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.whisper_attention import WhisperDecoderSelfAttention, WhisperCrossAttention
from ..L2.whisper_mlp import WhisperMLP


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = WhisperDecoderSelfAttention(
            config.d_model, config.decoder_attention_heads,
        )
        self.self_attn_layer_norm = LayerNorm(config.d_model, eps=1e-5)
        self.encoder_attn = WhisperCrossAttention(
            config.d_model, config.decoder_attention_heads,
        )
        self.encoder_attn_layer_norm = LayerNorm(config.d_model, eps=1e-5)
        self.mlp = WhisperMLP(config.d_model, config.decoder_ffn_dim)
        self.final_layer_norm = LayerNorm(config.d_model, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [N, D] flat token embeddings
            encoder_hidden_states: [N_enc, D] flat encoder outputs for NEW
                requests, or None when all requests are in decode phase.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states, encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
