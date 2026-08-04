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
        # promote_fp32=False to match vLLM, whose WhisperDecoderLayer uses a
        # plain ``nn.LayerNorm`` (model_executor/models/whisper.py). ATen's
        # layer_norm already accumulates its statistics in fp32 for a bf16
        # input, so promoting here only adds an x.float() before and a
        # .to(bf16) after -- three kernels per norm instead of one, at 96 norm
        # sites per decode step, plus an fp32 intermediate at double the
        # bandwidth. It also diverges from the reference, which applies the
        # affine weight in bf16 rather than in fp32 and then rounding.
        self.self_attn = WhisperDecoderSelfAttention(
            config.d_model, config.decoder_attention_heads,
        )
        self.self_attn_layer_norm = LayerNorm(
            config.d_model, eps=1e-5, promote_fp32=False,
        )
        self.encoder_attn = WhisperCrossAttention(
            config.d_model, config.decoder_attention_heads,
        )
        self.encoder_attn_layer_norm = LayerNorm(
            config.d_model, eps=1e-5, promote_fp32=False,
        )
        self.mlp = WhisperMLP(config.d_model, config.decoder_ffn_dim)
        self.final_layer_norm = LayerNorm(
            config.d_model, eps=1e-5, promote_fp32=False,
        )

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
