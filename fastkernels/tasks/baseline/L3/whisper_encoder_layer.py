"""Whisper encoder layer: self-attention + MLP with pre-norm LayerNorm.

Matches vLLM's WhisperEncoderLayer. Uses pre-norm residual connections:
  x = x + self_attn(layer_norm(x))
  x = x + mlp(layer_norm(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.whisper_attention import WhisperEncoderSelfAttention
from ..L2.whisper_mlp import WhisperMLP


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = WhisperEncoderSelfAttention(
            config.d_model, config.encoder_attention_heads,
        )
        self.self_attn_layer_norm = LayerNorm(config.d_model, eps=1e-5)
        self.mlp = WhisperMLP(config.d_model, config.encoder_ffn_dim)
        self.final_layer_norm = LayerNorm(config.d_model, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D]
        Returns:
            [B, T, D]
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states
