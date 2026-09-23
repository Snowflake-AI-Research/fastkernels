"""Blenderbot conditional generation with pre-norm blocks and learned positions."""

from fastkernels.hf_coverage.models.mbart import (
    PreNormConditionalGeneration,
    PreNormEncoderAttention,
    load_state_dict_into,
    make_workloads,
)
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("Blenderbot coverage requires tensor parallel size 1")
    if (config.activation_function != "gelu" or not config.tie_word_embeddings
            or not config.scale_embedding or not config.use_cache or not config.is_encoder_decoder):
        raise ValueError("Blenderbot requires GELU, scaled tied embeddings and default encoder-decoder caching")
    if config.d_model % config.encoder_attention_heads or config.d_model % config.decoder_attention_heads:
        raise ValueError("Blenderbot requires integral encoder and decoder head dimensions")

    # This carrier's zero-offset path has no embedding norm. Blenderbot's
    # position tables are learned, so retain their trainable weight identity.
    model = PreNormConditionalGeneration(config, learned_positions=False)
    # Importing the language dependencies disables cuDNN SDPA globally.
    # Preserve the isolated native reference's backend through existing ops.
    for layer in model.encoder.layers:
        layer.attn = PreNormEncoderAttention(layer.attn)
    for layer in model.decoder.layers:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
        layer.cross_attention.attention = DenseAttention(backend="cudnn")
    for stack in (model.encoder, model.decoder):
        stack.embed_positions.emb.weight.requires_grad_(True)
    return model.to(device=device, dtype=dtype).eval()
