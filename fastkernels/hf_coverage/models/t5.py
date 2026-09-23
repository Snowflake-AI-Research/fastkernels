"""T5 conditional generation with the existing encoder and decoder operation assembly."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_continuation_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from fastkernels.tasks.baseline.L3.t5_block import T5LayerFF
from fastkernels.tasks.baseline.L4.t5_encoder import T5Stack


def _fresh_cache(key, value):
    # HF DynamicLayer's first update copies K/V by concatenating empty caches.
    # Retain equivalent contiguous copies until the complete forward returns.
    return (key.clone(memory_format=torch.contiguous_format),
            value.clone(memory_format=torch.contiguous_format))


class T5CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.num_heads, config.d_kv
        width = self.heads * self.head_dim
        self.q = Linear(config.d_model, width, bias=False)
        self.k = Linear(config.d_model, width, bias=False)
        self.v = Linear(config.d_model, width, bias=False)
        self.o = Linear(width, config.d_model, bias=False)
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, hidden, memory, position_bias, past_key_value=None):
        batch, length = hidden.shape[:2]
        query = self.q(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        if past_key_value is None:
            key = self.k(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
            value = self.v(memory).view(batch, memory.shape[1], self.heads, self.head_dim).transpose(1, 2)
            cache = _fresh_cache(key, value)
        else:
            cache = past_key_value
        key, value = cache
        scores = self.bmm(query, key.transpose(2, 3)) + position_bias
        probabilities = self.softmax(scores.float()).to(scores.dtype)
        context = self.bmm(probabilities, value).transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o(context), cache


class T5DecoderBlock(nn.Module):
    def __init__(self, config, first):
        super().__init__()
        self.self_attention = T5SelfAttention(config, has_relative_attention_bias=first)
        self.self_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.cross_attention = T5CrossAttention(config)
        self.cross_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ff = T5LayerFF(config)

    def forward(self, hidden, memory, self_bias, cross_bias, past_key_value=None):
        normed = self.self_norm(hidden)
        attention = self.self_attention
        batch, length = normed.shape[:2]
        width = attention.n_heads_per_partition * attention.d_kv
        query, key, value = (
            tensor.view(batch, length, attention.n_heads_per_partition, attention.d_kv).transpose(1, 2)
            for tensor in attention.qkv_proj(normed).split(width, dim=-1)
        )
        if past_key_value is None:
            self_cache = _fresh_cache(key, value)
        else:
            self_cache = tuple(torch.cat((previous, current), dim=2)
                               for previous, current in zip(past_key_value[0], (key, value)))
        key, value = self_cache
        scores = attention.bmm(query, key.transpose(2, 3)) + self_bias
        probabilities = attention.softmax(scores.float()).to(scores.dtype)
        context = attention.bmm(probabilities, value).transpose(1, 2).contiguous().view(batch, length, -1)
        hidden = hidden + attention.o(context)
        cross_output, cross_cache = self.cross_attention(
            self.cross_norm(hidden), memory, cross_bias,
            None if past_key_value is None else past_key_value[1],
        )
        hidden = self.ff(hidden + cross_output)
        return hidden, (self_cache, cross_cache)


class T5ForConditionalGeneration(nn.Module):
    def __init__(self, config, *, output_scale=None, decoder_bias_per_layer=False):
        super().__init__()
        self.config = config
        self.shared = Embedding(config.vocab_size, config.d_model)
        self.encoder = self._build_encoder(config)
        self.decoder_bias_per_layer = decoder_bias_per_layer
        self.decoder = nn.ModuleList([
            T5DecoderBlock(config, first=decoder_bias_per_layer or index == 0)
            for index in range(config.num_decoder_layers)
        ])
        self.decoder_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.emb.weight
        if output_scale is None:
            output_scale = config.d_model ** -0.5 if config.scale_decoder_outputs else 1.0
        self.output_scale = output_scale

    def _build_encoder(self, config):
        return T5Stack(config, self.shared)

    def forward(self, input_ids, decoder_input_ids, buckets=None, causal_mask=None, *,
                encoder_hidden_states=None, past_key_values=None, attention_mask=None,
                decoder_attention_mask=None):
        memory = encoder_hidden_states
        if memory is None:
            memory = (self.encoder(input_ids) if attention_mask is None
                      else self.encoder(input_ids, attention_mask=attention_mask))
        hidden = self.shared(decoder_input_ids)
        length = decoder_input_ids.shape[1]
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        if buckets is None or causal_mask is None:
            query_positions = torch.arange(length, device=hidden.device) + past_length
            key_positions = torch.arange(past_length + length, device=hidden.device)
            if buckets is None:
                buckets = T5SelfAttention._relative_position_bucket(
                    key_positions[None, :] - query_positions[:, None], bidirectional=False,
                    num_buckets=self.config.relative_attention_num_buckets,
                    max_distance=self.config.relative_attention_max_distance,
                )
            if causal_mask is None:
                allowed = key_positions[None, :] <= query_positions[:, None]
                allowed = allowed[None, None, :, :]
                if decoder_attention_mask is not None:
                    allowed = allowed & decoder_attention_mask[:, None, None, :].bool()
                causal_mask = torch.zeros(allowed.shape, device=hidden.device, dtype=hidden.dtype)
                causal_mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        values = self.decoder[0].self_attention.relative_attention_bias(buckets)
        self_bias = values.permute(2, 0, 1).unsqueeze(0) + causal_mask
        cross_bias = torch.zeros(
            (1, values.shape[-1], length, memory.shape[1]),
            device=hidden.device, dtype=hidden.dtype,
        )
        if attention_mask is not None:
            cross_bias = cross_bias.expand(hidden.shape[0], -1, -1, -1).clone()
            cross_bias.masked_fill_(~attention_mask[:, None, None, :].bool(), torch.finfo(hidden.dtype).min)
        cache = []
        for index, block in enumerate(self.decoder):
            if index and self.decoder_bias_per_layer:
                values = block.self_attention.relative_attention_bias(buckets)
                self_bias = values.permute(2, 0, 1).unsqueeze(0) + causal_mask
            if past_key_values is None:
                hidden, layer_cache = block(hidden, memory, self_bias, cross_bias)
            else:
                hidden, layer_cache = block(hidden, memory, self_bias, cross_bias, past_key_values[index])
            cache.append(layer_cache)
        hidden = self.decoder_norm(hidden)
        if self.output_scale != 1.0:
            hidden = hidden * self.output_scale
        return {"logits": self.lm_head(hidden), "encoder_last_hidden_state": memory,
                "past_key_values": tuple(cache)}


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("T5 coverage requires tensor parallel size 1")
    if (config.feed_forward_proj != "relu" or config.is_gated_act
            or not config.tie_word_embeddings or not config.scale_decoder_outputs
            or not config.use_cache or not config.is_encoder_decoder):
        raise ValueError("T5-small requires ungated ReLU, tied scaled output and ordinary encoder-decoder caching")
    if config.num_heads * config.d_kv != config.d_model:
        raise ValueError("T5-small preserves equal attention and model widths")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("T5 coverage currently checks FP32 and BF16 loading semantics")
    return T5ForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    consumed = set()

    def copy(parameter, name):
        source = state_dict[name]
        if source.shape != parameter.shape:
            raise ValueError(f"T5 state shape mismatch: {name}")
        parameter.copy_(source)
        consumed.add(name)

    for name in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"):
        if not torch.equal(state_dict[name], state_dict["shared.weight"]):
            raise ValueError(f"T5 requires tied shared embeddings: {name}")
        consumed.add(name)
    model.shared.emb.weight.copy_(state_dict["shared.weight"])

    def self_attention(attention, prefix):
        packed = attention.qkv_proj.weight
        for shard in ("q", "k", "v"):
            name = prefix + shard + ".weight"
            source = state_dict[name]
            if source.shape != (config.num_heads * config.d_kv, config.d_model):
                raise ValueError(f"T5 QKV state shape mismatch: {name}")
            packed.weight_loader(packed, source, shard)
            consumed.add(name)
        copy(attention.o.weight, prefix + "o.weight")
        if attention.has_relative_attention_bias:
            copy(attention.relative_attention_bias.emb.weight, prefix + "relative_attention_bias.weight")

    def feed_forward(ff, prefix):
        copy(ff.layer_norm.weight, prefix + "layer_norm.weight")
        dense = ff.DenseReluDense
        if config.is_gated_act:
            for shard, name in enumerate(("wi_0", "wi_1")):
                source_name = prefix + f"DenseReluDense.{name}.weight"
                source = state_dict[source_name]
                if source.shape != (config.d_ff, config.d_model):
                    raise ValueError(f"T5 gated projection shape mismatch: {source_name}")
                dense.wi.weight.weight_loader(dense.wi.weight, source, shard)
                consumed.add(source_name)
        else:
            copy(dense.wi.weight, prefix + "DenseReluDense.wi.weight")
        copy(dense.wo.weight, prefix + "DenseReluDense.wo.weight")

    for index, block in enumerate(model.encoder.block):
        prefix = f"encoder.block.{index}.layer."
        self_attention(block.layer[0].SelfAttention, prefix + "0.SelfAttention.")
        copy(block.layer[0].layer_norm.weight, prefix + "0.layer_norm.weight")
        feed_forward(block.layer[1], prefix + "1.")
    copy(model.encoder.final_layer_norm.weight, "encoder.final_layer_norm.weight")
    for index, block in enumerate(model.decoder):
        prefix = f"decoder.block.{index}.layer."
        self_attention(block.self_attention, prefix + "0.SelfAttention.")
        copy(block.self_norm.weight, prefix + "0.layer_norm.weight")
        copy(block.cross_norm.weight, prefix + "1.layer_norm.weight")
        for name in ("q", "k", "v", "o"):
            copy(getattr(block.cross_attention, name).weight, prefix + f"1.EncDecAttention.{name}.weight")
        feed_forward(block.ff, prefix + "2.")
    copy(model.decoder_norm.weight, "decoder.final_layer_norm.weight")
    if consumed != set(state_dict):
        raise KeyError(f"T5 state has unconsumed tensors: {sorted(set(state_dict) - consumed)}")


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs)
    encoder_ids, decoder_ids = inputs["input_ids"], inputs["decoder_input_ids"]
    if encoder_ids.ndim != 2 or decoder_ids.ndim != 2 or encoder_ids.shape[0] != decoder_ids.shape[0]:
        raise ValueError("T5 requires matching encoder and decoder batches")
    length = decoder_ids.shape[1]
    positions = torch.arange(length, device=decoder_ids.device)
    buckets = T5SelfAttention._relative_position_bucket(
        positions[None, :] - positions[:, None], bidirectional=False,
        num_buckets=config.relative_attention_num_buckets,
        max_distance=config.relative_attention_max_distance,
    )
    causal_mask = torch.zeros((1, 1, length, length), device=decoder_ids.device, dtype=next(model.parameters()).dtype)
    causal_mask.masked_fill_(positions[None, :] > positions[:, None], torch.finfo(causal_mask.dtype).min)

    def run():
        output = model(encoder_ids, decoder_ids, buckets, causal_mask)
        return {name: output[name] for name in ("logits", "encoder_last_hidden_state")}

    return {"forward": Workload(run=run)}
