"""mBART conditional generation with pre-norm blocks and learned positions."""

import math

import torch
from torch import nn

from fastkernels.hf_coverage.models.bart import (
    BartCrossAttention, BartDecoderLayer, BartForConditionalGeneration,
    _fresh_cache, _cached_self_attention, make_workloads,
)
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock


class PreNormEncoderAttention(nn.Module):
    """Retain the existing projections and select native HF's SDPA backend."""

    def __init__(self, attention):
        super().__init__()
        self.qkv, self.proj = attention.qkv, attention.proj
        self.num_heads, self.head_dim = attention.num_heads, attention.head_dim
        self.attention = DenseAttention(backend="cudnn")

    def forward(self, hidden, attn_mask=None):
        shape = (*hidden.shape[:2], 3, self.num_heads, self.head_dim)
        query, key, value = self.qkv(hidden).reshape(shape).unbind(dim=2)
        context = self.attention(query, key, value, attn_mask=attn_mask)
        return self.proj(context.reshape_as(hidden))



class PreNormCrossAttention(BartCrossAttention):
    def forward(self, hidden, memory, past_key_value=None):
        normed = self.norm(hidden)
        batch, length = hidden.shape[:2]
        query = self.q_proj(normed).view(batch, length, self.heads, self.head_dim)
        if past_key_value is None:
            key = self.k_proj(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
            cache = _fresh_cache(key, value)
        else:
            cache = past_key_value
        key, value = cache
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2))
        return hidden + self.out_proj(context.reshape(batch, length, -1)), cache


class PreNormDecoderLayer(BartDecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.cross_attention = PreNormCrossAttention(config)
        if config.activation_function == "relu":
            self.intermediate.intermediate_act_fn = ReLU()

    def forward(self, hidden, memory, past_key_value=None):
        normed = self.attention.output.LayerNorm(hidden)
        attention = self.attention.self
        context, self_cache = _cached_self_attention(
            attention, normed, None if past_key_value is None else past_key_value[0],
        )
        hidden = hidden + self.attention.output.dense(context)
        hidden, cross_cache = self.cross_attention(
            hidden, memory, None if past_key_value is None else past_key_value[1],
        )
        hidden = hidden + self.output.dense(self.intermediate(self.output.LayerNorm(hidden)))
        return hidden, (self_cache, cross_cache)


class PreNormStack(nn.Module):
    def __init__(self, config, shared, *, decoder, learned_positions):
        super().__init__()
        self.embed_tokens = shared
        self.position_offset = 2 if learned_positions else 0
        self.embed_positions = Embedding(config.max_position_embeddings + self.position_offset, config.d_model)
        self.embed_positions.emb.weight.requires_grad_(learned_positions)
        self.layernorm_embedding = (LayerNorm(config.d_model, eps=1e-5, promote_fp32=False)
                                    if learned_positions else nn.Identity())
        self.layer_norm = LayerNorm(config.d_model, eps=1e-5, promote_fp32=False)
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.is_decoder = decoder
        if decoder:
            self.layers = nn.ModuleList([PreNormDecoderLayer(config) for _ in range(config.decoder_layers)])
        else:
            self.layers = nn.ModuleList([
                VitEncoderBlock(
                    config.d_model, config.encoder_attention_heads,
                    mlp_ratio=config.encoder_ffn_dim / config.d_model,
                    qkv_bias=True, proj_bias=True, act_approximate="none",
                    norm_eps=1e-5, attn_drop=config.attention_dropout, proj_drop=config.dropout,
                ) for _ in range(config.encoder_layers)
            ])
            if config.activation_function == "relu":
                for layer in self.layers:
                    layer.mlp.act = ReLU()

    def forward(self, ids, positions, memory=None, past_key_values=None):
        hidden = self.layernorm_embedding(self.embed_tokens(ids) * self.embed_scale + self.embed_positions(positions))
        cache = []
        for index, layer in enumerate(self.layers):
            if self.is_decoder:
                previous = None if past_key_values is None else past_key_values[index]
                hidden, layer_cache = layer(hidden, memory, previous)
                cache.append(layer_cache)
            else:
                hidden = layer(hidden)
        hidden = self.layer_norm(hidden)
        return (hidden, tuple(cache)) if self.is_decoder else hidden


class PreNormConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config, *, learned_positions):
        nn.Module.__init__(self)
        self.shared = Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.encoder = PreNormStack(config, self.shared, decoder=False, learned_positions=learned_positions)
        self.decoder = PreNormStack(config, self.shared, decoder=True, learned_positions=learned_positions)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.emb.weight
        self.register_buffer("final_logits_bias", torch.zeros(1, config.vocab_size))


def select_native_attention(model):
    """Keep native cuDNN SDPA available despite vLLM's import-time global toggle."""
    for layer in model.encoder.layers:
        layer.attn = PreNormEncoderAttention(layer.attn)
    for layer in model.decoder.layers:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
        layer.cross_attention.attention = DenseAttention(backend="cudnn")


def build_pre_norm(config, device, dtype, *, learned_positions):
    if _tp_size() != 1:
        raise ValueError("Pre-norm seq2seq coverage requires tensor parallel size 1")
    activation = "gelu" if learned_positions else "relu"
    if (config.activation_function != activation or not config.tie_word_embeddings
            or not config.scale_embedding or not config.use_cache or not config.is_encoder_decoder):
        raise ValueError("Selected checkpoint requires its activation, scaled tied embeddings and default seq2seq cache")
    if config.d_model % config.encoder_attention_heads or config.d_model % config.decoder_attention_heads:
        raise ValueError("Encoder and decoder head dimensions must be integral")
    model = PreNormConditionalGeneration(config, learned_positions=learned_positions)
    select_native_attention(model)
    return model.to(device=device, dtype=dtype).eval()


def build_from_config(config, device, dtype):
    return build_pre_norm(config, device, dtype, learned_positions=True)


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    consumed = set()

    def copy(parameter, name):
        source = state_dict[name]
        if source.shape != parameter.shape:
            raise ValueError(f"Pre-norm seq2seq state shape mismatch: {name}")
        parameter.copy_(source)
        consumed.add(name)

    def affine(module, prefix):
        copy(module.weight, prefix + "weight")
        copy(module.bias, prefix + "bias")

    def qkv(module, prefix):
        for suffix in ("weight", "bias"):
            names = [prefix + f"{shard}_proj.{suffix}" for shard in ("q", "k", "v")]
            source = torch.cat([state_dict[name] for name in names], dim=0)
            parameter = getattr(module, suffix)
            if parameter.shape != source.shape:
                raise ValueError(f"Pre-norm seq2seq packed QKV mismatch: {prefix}{suffix}")
            parameter.copy_(source)
            consumed.update(names)

    for name in ("model.shared.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"):
        if not torch.equal(state_dict[name], state_dict["model.shared.weight"]):
            raise ValueError(f"Pre-norm seq2seq requires tied shared embeddings: {name}")
        consumed.add(name)
    model.shared.emb.weight.copy_(state_dict["model.shared.weight"])
    copy(model.final_logits_bias, "final_logits_bias")
    for stack_name, stack in (("encoder", model.encoder), ("decoder", model.decoder)):
        prefix = f"model.{stack_name}."
        # Pegasus's frozen sinusoidal table is persistent HF state. Load that
        # exact shared table through the existing embedding lookup operation.
        copy(stack.embed_positions.emb.weight, prefix + "embed_positions.weight")
        if stack.position_offset:
            affine(stack.layernorm_embedding, prefix + "layernorm_embedding.")
        affine(stack.layer_norm, prefix + "layer_norm.")
        for index, layer in enumerate(stack.layers):
            layer_prefix = prefix + f"layers.{index}."
            if stack.is_decoder:
                qkv(layer.attention.self.qkv, layer_prefix + "self_attn.")
                affine(layer.attention.output.dense, layer_prefix + "self_attn.out_proj.")
                affine(layer.attention.output.LayerNorm, layer_prefix + "self_attn_layer_norm.")
                affine(layer.intermediate.dense, layer_prefix + "fc1.")
                affine(layer.output.dense, layer_prefix + "fc2.")
                affine(layer.output.LayerNorm, layer_prefix + "final_layer_norm.")
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    affine(getattr(layer.cross_attention, name), layer_prefix + f"encoder_attn.{name}.")
                affine(layer.cross_attention.norm, layer_prefix + "encoder_attn_layer_norm.")
            else:
                qkv(layer.attn.qkv, layer_prefix + "self_attn.")
                affine(layer.attn.proj, layer_prefix + "self_attn.out_proj.")
                affine(layer.norm1, layer_prefix + "self_attn_layer_norm.")
                affine(layer.mlp.fc1, layer_prefix + "fc1.")
                affine(layer.mlp.fc2, layer_prefix + "fc2.")
                affine(layer.norm2, layer_prefix + "final_layer_norm.")
    if consumed != set(state_dict):
        raise KeyError(f"Pre-norm seq2seq has unconsumed state: {sorted(set(state_dict) - consumed)}")
