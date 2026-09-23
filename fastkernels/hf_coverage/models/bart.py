"""BART conditional generation with post-norm encoder and cached decoder assembly."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_continuation_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer


def _carrier(config, decoder):
    return SimpleNamespace(
        hidden_size=config.d_model, layer_norm_eps=1e-5,
        intermediate_size=config.decoder_ffn_dim if decoder else config.encoder_ffn_dim,
        num_attention_heads=config.decoder_attention_heads if decoder else config.encoder_attention_heads,
        num_hidden_layers=config.decoder_layers if decoder else config.encoder_layers,
    )


def _fresh_cache(key, value):
    # Match the first DynamicLayer update's contiguous K/V copies and retention.
    return (key.clone(memory_format=torch.contiguous_format),
            value.clone(memory_format=torch.contiguous_format))


def _cached_self_attention(attention, hidden, past=None):
    batch, length = hidden.shape[:2]
    query, key, value = (
        tensor.view(batch, length, attention.num_attention_heads, attention.attention_head_size)
        for tensor in attention._project_qkv(hidden)
    )
    key, value = key.transpose(1, 2), value.transpose(1, 2)
    if past is None:
        cache = _fresh_cache(key, value)
        kwargs = {"causal": True}
    else:
        cache = tuple(torch.cat((old, new), dim=2) for old, new in zip(past, (key, value)))
        # A single new token can attend to the complete cache. For a chunk,
        # align the causal boundary with its position after the previous tokens.
        kwargs = {}
        if length > 1:
            query_positions = torch.arange(length, device=hidden.device) + past[0].shape[2]
            key_positions = torch.arange(cache[0].shape[2], device=hidden.device)
            mask = torch.zeros((length, cache[0].shape[2]), device=hidden.device, dtype=hidden.dtype)
            mask.masked_fill_(key_positions[None, :] > query_positions[:, None], torch.finfo(hidden.dtype).min)
            kwargs["attn_mask"] = mask
    key, value = cache
    context = attention.attn(query, key.transpose(1, 2), value.transpose(1, 2), **kwargs)
    return context.reshape(batch, length, -1), cache


class BartCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.decoder_attention_heads
        self.head_dim = config.d_model // self.heads
        self.q_proj = Linear(config.d_model, config.d_model)
        self.k_proj = Linear(config.d_model, config.d_model)
        self.v_proj = Linear(config.d_model, config.d_model)
        self.out_proj = Linear(config.d_model, config.d_model)
        self.attention = DenseAttention(backend="sdpa")
        self.norm = LayerNorm(config.d_model, eps=1e-5, promote_fp32=False)

    def forward(self, hidden, memory, past_key_value=None):
        batch, length = hidden.shape[:2]
        query = self.q_proj(hidden).view(batch, length, self.heads, self.head_dim)
        if past_key_value is None:
            key = self.k_proj(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
            cache = _fresh_cache(key, value)
        else:
            cache = past_key_value
        key, value = cache
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2))
        hidden = self.norm(hidden + self.out_proj(context.reshape(batch, length, -1)))
        return hidden, cache


class BartDecoderLayer(BertLayer):
    def __init__(self, config):
        super().__init__(_carrier(config, decoder=True))
        self.cross_attention = BartCrossAttention(config)

    def forward(self, hidden, memory, past_key_value=None):
        attention = self.attention.self
        context, self_cache = _cached_self_attention(
            attention, hidden, None if past_key_value is None else past_key_value[0],
        )
        hidden = self.attention.output(context, hidden)
        hidden, cross_cache = self.cross_attention(
            hidden, memory, None if past_key_value is None else past_key_value[1],
        )
        hidden = self.output(self.intermediate(hidden), hidden)
        return hidden, (self_cache, cross_cache)


class BartStack(nn.Module):
    def __init__(self, config, shared, decoder):
        super().__init__()
        self.embed_tokens = shared
        self.position_offset = 2
        self.embed_positions = Embedding(config.max_position_embeddings + 2, config.d_model)
        self.layernorm_embedding = LayerNorm(config.d_model, eps=1e-5, promote_fp32=False)
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.is_decoder = decoder
        self.layers = (nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
                       if decoder else BertEncoder(_carrier(config, decoder=False)))

    def forward(self, ids, positions, memory=None, past_key_values=None):
        hidden = self.embed_tokens(ids) * self.embed_scale
        hidden = self.layernorm_embedding(hidden + self.embed_positions(positions))
        if not self.is_decoder:
            return self.layers(hidden)
        cache = []
        for index, layer in enumerate(self.layers):
            previous = None if past_key_values is None else past_key_values[index]
            hidden, layer_cache = layer(hidden, memory, previous)
            cache.append(layer_cache)
        return hidden, tuple(cache)


class BartForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shared = Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.encoder = BartStack(config, self.shared, decoder=False)
        self.decoder = BartStack(config, self.shared, decoder=True)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.emb.weight
        self.register_buffer("final_logits_bias", torch.zeros(1, config.vocab_size))

    def forward(self, input_ids, decoder_input_ids, encoder_positions=None, decoder_positions=None, *,
                encoder_hidden_states=None, past_key_values=None, attention_mask=None,
                decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("BART coverage currently evaluates unpadded token sequences")
        memory = encoder_hidden_states
        if memory is None:
            if encoder_positions is None:
                encoder_positions = torch.arange(input_ids.shape[1], device=input_ids.device) + self.encoder.position_offset
            memory = self.encoder(input_ids, encoder_positions)
        if decoder_positions is None:
            past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
            decoder_positions = (torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)
                                 + past_length + self.decoder.position_offset)
        hidden, cache = self.decoder(decoder_input_ids, decoder_positions, memory, past_key_values)
        return {"logits": self.lm_head(hidden) + self.final_logits_bias,
                "encoder_last_hidden_state": memory, "past_key_values": cache}


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("BART coverage requires tensor parallel size 1")
    if (config.activation_function != "gelu" or not config.tie_word_embeddings
            or not config.use_cache or not config.is_encoder_decoder
            or getattr(config, "normalize_before", False) or getattr(config, "add_final_layer_norm", False)):
        raise ValueError("BART-large-CNN requires post-norm GELU, tied embeddings and ordinary encoder-decoder caching")
    if config.d_model % config.encoder_attention_heads or config.d_model % config.decoder_attention_heads:
        raise ValueError("BART requires integral encoder and decoder head dimensions")
    model = BartForConditionalGeneration(config)
    # Native HF SDPA uses cuDNN on the evaluated B200. FastKernels' vLLM
    # imports disable it globally; select the existing backend explicitly.
    for layer in model.encoder.layers.layer:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
    for layer in model.decoder.layers:
        layer.attention.self.attn = DenseAttention(backend="cudnn")
        layer.cross_attention.attention = DenseAttention(backend="cudnn")
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    consumed = set()

    def copy(parameter, name):
        source = state_dict[name]
        if source.shape != parameter.shape:
            raise ValueError(f"BART state shape mismatch: {name}")
        parameter.copy_(source)
        consumed.add(name)

    def affine(module, prefix):
        copy(module.weight, prefix + "weight")
        copy(module.bias, prefix + "bias")

    def self_attention(attention, prefix):
        for suffix in ("weight", "bias"):
            names = [prefix + f"{shard}_proj.{suffix}" for shard in ("q", "k", "v")]
            source = torch.cat([state_dict[name] for name in names], dim=0)
            parameter = getattr(attention.self.qkv, suffix)
            if parameter.shape != source.shape:
                raise ValueError(f"BART packed QKV shape mismatch: {prefix}{suffix}")
            parameter.copy_(source)
            consumed.update(names)
        affine(attention.output.dense, prefix + "out_proj.")

    for name in ("model.shared.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"):
        if not torch.equal(state_dict[name], state_dict["model.shared.weight"]):
            raise ValueError(f"BART requires tied shared embeddings: {name}")
        consumed.add(name)
    model.shared.emb.weight.copy_(state_dict["model.shared.weight"])
    copy(model.final_logits_bias, "final_logits_bias")
    for stack_name, stack in (("encoder", model.encoder), ("decoder", model.decoder)):
        prefix = f"model.{stack_name}."
        copy(stack.embed_positions.emb.weight, prefix + "embed_positions.weight")
        affine(stack.layernorm_embedding, prefix + "layernorm_embedding.")
        layers = stack.layers if stack.is_decoder else stack.layers.layer
        for index, layer in enumerate(layers):
            layer_prefix = prefix + f"layers.{index}."
            self_attention(layer.attention, layer_prefix + "self_attn.")
            affine(layer.attention.output.LayerNorm, layer_prefix + "self_attn_layer_norm.")
            affine(layer.intermediate.dense, layer_prefix + "fc1.")
            affine(layer.output.dense, layer_prefix + "fc2.")
            affine(layer.output.LayerNorm, layer_prefix + "final_layer_norm.")
            if stack.is_decoder:
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    affine(getattr(layer.cross_attention, name), layer_prefix + f"encoder_attn.{name}.")
                affine(layer.cross_attention.norm, layer_prefix + "encoder_attn_layer_norm.")
    if consumed != set(state_dict):
        raise KeyError(f"BART state has unconsumed tensors: {sorted(set(state_dict) - consumed)}")


def make_workloads(model, inputs, config, *, case=None):
    encoder_ids, decoder_ids = inputs["input_ids"], inputs["decoder_input_ids"]
    if encoder_ids.ndim != 2 or decoder_ids.ndim != 2 or encoder_ids.shape[0] != decoder_ids.shape[0]:
        raise ValueError("BART requires matching encoder and decoder batches")
    if max(encoder_ids.shape[1], decoder_ids.shape[1]) > config.max_position_embeddings:
        raise ValueError("BART input exceeds its learned position table")
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs)
    encoder_positions = torch.arange(encoder_ids.shape[1], device=encoder_ids.device) + model.encoder.position_offset
    decoder_positions = torch.arange(decoder_ids.shape[1], device=decoder_ids.device) + model.decoder.position_offset

    def run():
        output = model(encoder_ids, decoder_ids, encoder_positions, decoder_positions)
        return {name: output[name] for name in ("logits", "encoder_last_hidden_state")}

    return {"forward": Workload(run=run)}
