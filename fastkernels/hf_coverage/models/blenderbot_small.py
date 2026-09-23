"""Blenderbot Small conditional generation with its distinct decoder embedding order."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.bart import (
    BartForConditionalGeneration,
    BartStack,
    load_state_dict_into,
)
from fastkernels.hf_coverage.models.mbart import make_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear


class BlenderbotSmallStack(BartStack):
    def __init__(self, config, shared, decoder):
        super().__init__(config, shared, decoder)
        self.position_offset = 0
        self.embed_positions = Embedding(config.max_position_embeddings, config.d_model)

    def forward(self, ids, positions, memory=None, past_key_values=None):
        if not self.is_decoder:
            return super().forward(ids, positions)
        # Pinned HF scales only the encoder embeddings. The decoder applies
        # its embedding norm before adding the unnormalized position vectors.
        hidden = self.layernorm_embedding(self.embed_tokens(ids)) + self.embed_positions(positions)
        cache = []
        for index, layer in enumerate(self.layers):
            previous = None if past_key_values is None else past_key_values[index]
            hidden, layer_cache = layer(hidden, memory, previous)
            cache.append(layer_cache)
        return hidden, tuple(cache)


class BlenderbotSmallForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.shared = Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.encoder = BlenderbotSmallStack(config, self.shared, decoder=False)
        self.decoder = BlenderbotSmallStack(config, self.shared, decoder=True)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.emb.weight
        self.register_buffer("final_logits_bias", torch.zeros(1, config.vocab_size))


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("Blenderbot Small coverage requires tensor parallel size 1")
    if (config.activation_function != "gelu" or not config.tie_word_embeddings
            or not config.scale_embedding or not config.use_cache or not config.is_encoder_decoder):
        raise ValueError("Blenderbot Small requires GELU, scaled encoder embeddings, tied weights and default caching")
    if config.d_model % config.encoder_attention_heads or config.d_model % config.decoder_attention_heads:
        raise ValueError("Blenderbot Small requires integral encoder and decoder head dimensions")
    model = BlenderbotSmallForConditionalGeneration(config)
    # This post-norm stack has the same native SDPA backend requirement as BART.
    for layer in model.encoder.layers.layer:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
    for layer in model.decoder.layers:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
        layer.cross_attention.attention = DenseAttention(backend="cudnn")
    return model.to(device=device, dtype=dtype).eval()
