"""Standalone Whisper encoder-decoder model for speech-to-text.

Implements the full WhisperForConditionalGeneration architecture:
  - Audio encoder: Conv1d feature extraction + sinusoidal pos embeddings + transformer layers
  - Text decoder: token embedding + learned pos embeddings + transformer layers + cross-attention
  - LM head (tied to decoder embedding)

Matches vLLM's whisper.py WhisperForConditionalGeneration interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig

from ..L1.conv1d import Conv1d
from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L2.parallel_embedding import ParallelLMHead
from ..L3.whisper_encoder_layer import WhisperEncoderLayer
from ..L3.whisper_decoder_layer import WhisperDecoderLayer


@dataclass
class WhisperConfig:
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    decoder_layers: int = 32
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    vocab_size: int = 51866
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    max_target_positions: int = 448
    scale_embedding: bool = False
    activation_function: str = "gelu"
    pad_token_id: int = 50256
    bos_token_id: int = 50257
    eos_token_id: int = 50257
    decoder_start_token_id: int = 50258
    is_encoder_decoder: bool = True
    dtype: torch.dtype = torch.bfloat16
    # For compatibility with engine's KV cache allocation
    num_hidden_layers: int = 32
    num_key_value_heads: int = 20
    num_attention_heads: int = 20
    head_dim: int = 64
    hidden_size: int = 1280

    @classmethod
    def from_pretrained(cls, model_name: str) -> "WhisperConfig":
        hf = AutoConfig.from_pretrained(model_name)
        head_dim = hf.d_model // hf.decoder_attention_heads
        return cls(
            d_model=hf.d_model,
            encoder_layers=hf.encoder_layers,
            encoder_attention_heads=hf.encoder_attention_heads,
            encoder_ffn_dim=hf.encoder_ffn_dim,
            decoder_layers=hf.decoder_layers,
            decoder_attention_heads=hf.decoder_attention_heads,
            decoder_ffn_dim=hf.decoder_ffn_dim,
            vocab_size=hf.vocab_size,
            num_mel_bins=hf.num_mel_bins,
            max_source_positions=hf.max_source_positions,
            max_target_positions=hf.max_target_positions,
            scale_embedding=hf.scale_embedding,
            activation_function=hf.activation_function,
            pad_token_id=hf.pad_token_id,
            bos_token_id=hf.bos_token_id,
            eos_token_id=hf.eos_token_id,
            decoder_start_token_id=hf.decoder_start_token_id,
            is_encoder_decoder=hf.is_encoder_decoder,
            num_hidden_layers=hf.decoder_layers,
            num_key_value_heads=hf.decoder_attention_heads,
            num_attention_heads=hf.decoder_attention_heads,
            head_dim=head_dim,
            hidden_size=hf.d_model,
        )


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions

        self.conv1 = Conv1d(config.num_mel_bins, embed_dim,
                            kernel_size=3, padding=1)
        self.conv2 = Conv1d(embed_dim, embed_dim,
                            kernel_size=3, stride=2, padding=1)
        self.gelu = GELU()

        self.layers = nn.ModuleList([
            WhisperEncoderLayer(config)
            for _ in range(config.encoder_layers)
        ])
        self.layer_norm = LayerNorm(embed_dim, eps=1e-5)

        self.embed_positions = Embedding(config.max_source_positions, embed_dim)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [B, num_mel_bins, T] log-mel spectrograms
        Returns:
            [B, T_enc, D] encoder hidden states
        """
        input_features = input_features.to(self.conv1.conv.weight.dtype)
        hidden_states = self.gelu(self.conv1(input_features))
        hidden_states = self.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(-1, -2)

        T_enc = hidden_states.shape[1]
        hidden_states = hidden_states + self.embed_positions.emb.weight[:T_enc].to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.max_target_positions = config.max_target_positions
        self.d_model = config.d_model

        self.embed_tokens = Embedding(config.vocab_size, config.d_model,
                                       padding_idx=config.pad_token_id)
        self.embed_positions = Embedding(config.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([
            WhisperDecoderLayer(config)
            for _ in range(config.decoder_layers)
        ])
        self.layer_norm = LayerNorm(config.d_model, eps=1e-5)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [N] flat token IDs
            positions: [N] position indices
            encoder_hidden_states: [N_enc, D] flat encoder outputs for NEW
                requests, or None when all are in decode phase.
        """
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        hidden_states = inputs_embeds + pos_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "self_attn.q_proj": ("self_attn.qkv_proj", "q"),
        "self_attn.k_proj": ("self_attn.qkv_proj", "k"),
        "self_attn.v_proj": ("self_attn.qkv_proj", "v"),
        "encoder_attn.k_proj": ("encoder_attn.kv_proj", "k"),
        "encoder_attn.v_proj": ("encoder_attn.kv_proj", "v"),
    }

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.d_model)
        self._linear_op = Linear()

    def get_multimodal_embeddings(
        self, input_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run encoder on audio features, return per-request encoder outputs.

        Args:
            input_features: [B, num_mel_bins, T] batched log-mel spectrograms
        Returns:
            List of [T_enc, D] tensors, one per batch element.
        """
        encoder_out = self.encoder(input_features)
        return list(encoder_out.unbind(0))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Decoder forward pass.

        Args:
            input_ids: [N] flat token IDs
            positions: [N] position indices
            encoder_outputs: list of [T_enc_i, D] encoder hidden states
                for NEW requests this step. Concatenated and passed to
                cross-attention for KV projection and cache write.
                Empty list or None means all requests are decoding
                from cached KV.
        """
        enc_states = None
        if encoder_outputs:
            enc_states = torch.cat(encoder_outputs, dim=0)

        return self.decoder(input_ids, positions, enc_states)

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if logits is not None:
            logits = logits.float()
        return logits

    def compute_logits_decode(self, partial_logits):
        logits = self.lm_head.gather_logits(partial_logits)
        if logits is not None:
            logits = logits.float()
        return logits

    def greedy_sample_decode(self, partial_logits):
        result = self.lm_head.gather_greedy(partial_logits.float())
        return result
