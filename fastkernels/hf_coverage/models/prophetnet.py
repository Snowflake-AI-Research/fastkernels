"""ProphetNet full conditional generation, including both prediction streams."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs, seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention
from .bart import _fresh_cache


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        for name in ("query_proj", "key_proj", "value_proj", "out_proj"):
            setattr(self, name, Linear(width, width))
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)

    def shape(self, tensor):
        return tensor.reshape(tensor.shape[0], -1, self.heads, self.dim).transpose(1, 2)

    def forward(self, hidden, memory=None, past_key_value=None):
        query = self.shape(self.query_proj(hidden) / self.dim ** 0.5)
        if past_key_value is not None:
            key, value = past_key_value
        else:
            source = hidden if memory is None else memory
            key, value = self.shape(self.key_proj(source)), self.shape(self.value_proj(source))
            if memory is not None:
                key, value = _fresh_cache(key, value)
        probabilities = self.softmax(self.bmm(query, key.transpose(-1, -2)))
        context = self.bmm(probabilities, value).transpose(1, 2).reshape_as(hidden)
        return self.out_proj(context), (key, value)


class NgramAttention(Attention):
    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_decoder_attention_heads)
        self.ngram, self.buckets = config.ngram, config.num_buckets
        self.relative_pos_embeddings = Linear(config.hidden_size, self.buckets * self.heads)

    def forward(self, hidden, main_buckets, predict_buckets, main_mask, predict_mask,
                past_key_value=None):
        batch, streams_length, width = hidden.shape
        length = streams_length // (1 + self.ngram)
        query = self.shape(self.query_proj(hidden) / self.dim ** 0.5).chunk(1 + self.ngram, dim=2)
        key = self.shape(self.key_proj(hidden)).chunk(1 + self.ngram, dim=2)
        value = self.shape(self.value_proj(hidden)).chunk(1 + self.ngram, dim=2)
        if past_key_value is None:
            main_key, main_value = cache = _fresh_cache(key[0], value[0])
        else:
            main_key, main_value = cache = tuple(
                torch.cat((old, new), dim=2) for old, new in zip(past_key_value, (key[0], value[0])))
        main_rel = self.relative_pos_embeddings(hidden[:, :length])
        main_rel = main_rel.view(batch, length, self.buckets, self.heads).permute(0, 3, 1, 2)
        main_index = main_buckets[:, None].expand(-1, self.heads, -1, -1)
        main_rel = main_rel.gather(-1, main_index)
        scores = self.bmm(query[0], main_key.transpose(-1, -2)) + main_rel
        if main_mask is not None:
            scores = scores + main_mask
        probabilities = self.softmax(scores.float()).to(hidden.dtype)
        context = self.bmm(probabilities, main_value).transpose(1, 2).reshape(batch, 1, length, width)
        main_output = self.out_proj(context)

        predict_query = torch.stack(query[1:], dim=1)
        predict_key = torch.stack([torch.cat((main_key, tensor), dim=2) for tensor in key[1:]], dim=1)
        predict_value = torch.stack([torch.cat((main_value, tensor), dim=2) for tensor in value[1:]], dim=1)
        predict_hidden = hidden[:, length:].reshape(batch, self.ngram, length, width)
        relative = self.relative_pos_embeddings(predict_hidden)
        # Preserve the pinned implementation's gather order across stream,
        # position and head axes, including its buffered right-position offset.
        relative = relative.view(batch, self.ngram, length, self.buckets, self.heads)
        relative = relative.permute(0, 2, 1, 4, 3).reshape(-1, self.buckets)
        key_length = main_key.shape[2] + length
        indices = predict_buckets[None].repeat(self.ngram, 1, self.heads, 1).reshape(-1, key_length)
        relative = relative.gather(1, indices).reshape(batch, self.ngram, self.heads, length, key_length)
        scores = self.bmm(predict_query, predict_key.transpose(-1, -2)) + relative
        if predict_mask is not None:
            scores = scores + predict_mask
        probabilities = self.softmax(scores.float()).to(hidden.dtype)
        context = self.bmm(probabilities, predict_value).transpose(2, 3).reshape(batch, self.ngram, length, width)
        predict_output = self.out_proj(context)
        return torch.cat((main_output, predict_output), dim=1).reshape_as(hidden), cache


class FeedForward(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.intermediate = Linear(width, intermediate)
        self.output = Linear(intermediate, width)
        self.activation = GELU()

    def forward(self, hidden):
        return self.output(self.activation(self.intermediate(hidden)))


class Layer(nn.Module):
    def __init__(self, config, decoder):
        super().__init__()
        self.decoder = decoder
        self.self_attn = (NgramAttention(config) if decoder else Attention(config.hidden_size, config.num_encoder_attention_heads))
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, promote_fp32=False)
        if decoder:
            self.cross_attn = Attention(config.hidden_size, config.num_decoder_attention_heads)
            self.cross_attn_layer_norm = LayerNorm(config.hidden_size, promote_fp32=False)
        intermediate = config.decoder_ffn_dim if decoder else config.encoder_ffn_dim
        self.feed_forward = FeedForward(config.hidden_size, intermediate)
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size, promote_fp32=False)

    def forward(self, hidden, memory=None, metadata=None, past_key_value=None):
        output, self_cache = (self.self_attn(hidden, **metadata,
            past_key_value=None if past_key_value is None else past_key_value[0])
            if self.decoder else self.self_attn(hidden))
        hidden = self.self_attn_layer_norm(hidden + output)
        if self.decoder:
            output, cross_cache = self.cross_attn(hidden, memory,
                None if past_key_value is None else past_key_value[1])
            hidden = self.cross_attn_layer_norm(hidden + output)
        hidden = self.feed_forward_layer_norm(hidden + self.feed_forward(hidden))
        return (hidden, (self_cache, cross_cache)) if self.decoder else hidden


class Stack(nn.Module):
    def __init__(self, config, shared, decoder):
        super().__init__()
        self.word_embeddings = shared
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size, config.pad_token_id)
        self.embeddings_layer_norm = LayerNorm(config.hidden_size, promote_fp32=False)
        if decoder:
            self.ngram_embeddings = Embedding(config.ngram, config.hidden_size)
        count = config.num_decoder_layers if decoder else config.num_encoder_layers
        self.layers = nn.ModuleList([Layer(config, decoder) for _ in range(count)])


class ProphetNetForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prophetnet = nn.Module()
        self.prophetnet.word_embeddings = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.prophetnet.encoder = Stack(config, self.prophetnet.word_embeddings, False)
        self.prophetnet.decoder = Stack(config, self.prophetnet.word_embeddings, True)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.prophetnet.word_embeddings.emb.weight
        self.ngram = config.ngram
        self.config = config

    def forward(self, input_ids, decoder_input_ids, encoder_positions=None, decoder_positions=None,
                metadata=None, *, encoder_hidden_states=None, past_key_values=None,
                attention_mask=None, decoder_attention_mask=None):
        if attention_mask is not None or decoder_attention_mask is not None:
            raise ValueError("ProphetNet coverage evaluates unpadded encoder and decoder inputs")
        encoder, decoder = self.prophetnet.encoder, self.prophetnet.decoder
        memory = encoder_hidden_states
        if memory is None:
            if encoder_positions is None:
                encoder_positions = (torch.arange(1, input_ids.shape[1] + 1, device=input_ids.device)
                                     + self.config.pad_token_id)[None].expand(input_ids.shape[0], -1)
            memory = encoder.embeddings_layer_norm(encoder.word_embeddings(input_ids) + encoder.position_embeddings(encoder_positions))
            for layer in encoder.layers:
                memory = layer(memory)
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        if past_length and decoder_input_ids.shape[1] != 1:
            raise ValueError("Native ProphetNet cached continuation requires one new token")
        if decoder_positions is None:
            decoder_positions = (torch.arange(1, decoder_input_ids.shape[1] + 1, device=decoder_input_ids.device)
                                 + past_length + self.config.pad_token_id)[None].expand(decoder_input_ids.shape[0], -1)
        if metadata is None:
            metadata = attention_metadata(decoder_input_ids, self.config, memory.dtype, past_length)
        hidden = decoder.word_embeddings(decoder_input_ids) + decoder.position_embeddings(decoder_positions)
        predicting_positions = decoder.position_embeddings(decoder_positions + 1)
        streams = [decoder.ngram_embeddings.emb.weight[index - 1] + predicting_positions for index in range(self.ngram)]
        hidden = decoder.embeddings_layer_norm(torch.cat((hidden, *streams), dim=1))
        cache = []
        for index, layer in enumerate(decoder.layers):
            previous = None if past_key_values is None else past_key_values[index]
            hidden, layer_cache = layer(hidden, memory, metadata, previous)
            cache.append(layer_cache)
        batch, length = decoder_input_ids.shape
        logits = self.lm_head(hidden[:, length:].reshape(batch, self.ngram, length, -1))
        return {"logits": logits[:, 0].contiguous(), "logits_ngram": logits[:, 1:],
                "encoder_last_hidden_state": memory, "past_key_values": tuple(cache)}


def build_from_config(config, device, dtype):
    if (config.activation_function != "gelu" or config.ngram != 2 or not config.add_cross_attention
            or not config.tie_word_embeddings or not config.use_cache):
        raise ValueError("Selected ProphetNet requires GELU, two prediction streams, cross-attention, tied output and caching")
    return ProphetNetForConditionalGeneration(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {name: state_dict[name.replace(".emb.weight", ".weight")] for name in model.state_dict()}
    consumed = {name.replace(".emb.weight", ".weight") for name in mapped}
    if consumed != set(state_dict):
        raise ValueError(f"ProphetNet unmapped state: {sorted(set(state_dict) - consumed)}")
    for name in ("prophetnet.encoder.word_embeddings.weight", "prophetnet.decoder.word_embeddings.weight", "lm_head.weight"):
        if not torch.equal(state_dict[name], state_dict["prophetnet.word_embeddings.weight"]):
            raise ValueError(f"ProphetNet requires tied token embeddings: {name}")
    model.load_state_dict(mapped)


def attention_metadata(ids, config, dtype, past_length=0):
    """Relative buckets and masks depend on positions, never on activations."""
    batch, length = ids.shape
    positions = torch.arange(1, length + 1, device=ids.device)

    def buckets(relative):
        return T5SelfAttention._relative_position_bucket(relative, bidirectional=False,
            num_buckets=config.num_buckets, max_distance=config.relative_max_distance)

    if past_length:
        # Native cached attention uses its unbuffered position buckets, and
        # the sole new query can see every retained main-stream key.
        current = past_length + length + config.pad_token_id
        main = torch.arange(1, past_length + length + 1, device=ids.device) - current
        prediction = torch.arange(past_length + 2 * length, device=ids.device) - current
        return dict(main_buckets=buckets(main)[None, None].expand(batch, -1, -1),
                    predict_buckets=buckets(prediction)[None, None].expand(batch, -1, -1),
                    main_mask=None, predict_mask=None)
    main_buckets = buckets(positions[None] - positions[:, None])[None].expand(batch, -1, -1)
    predict_buckets = buckets(torch.cat((positions - 1, positions + 1))[None] - positions[:, None])[None].expand(batch, -1, -1)
    minimum = torch.finfo(dtype).min
    main_mask = torch.full((length, length), minimum, device=ids.device, dtype=dtype).triu(1)[None, None]
    left = torch.full((config.ngram, length, length), minimum, device=ids.device, dtype=dtype)
    right = torch.full_like(left, minimum)
    for stream in range(config.ngram):
        left[stream].triu_(1 - stream)
        right[stream].fill_diagonal_(0)
    left[:, :, 0] = 0
    predict_mask = torch.cat((left, right), dim=-1)[None, :, None]
    return dict(main_buckets=main_buckets, predict_buckets=predict_buckets,
                main_mask=main_mask, predict_mask=predict_mask)


def make_workloads(model, inputs, config, *, case=None):
    if case is not None and case["workload"] == "seq2seq_continuation":
        return seq2seq_continuation_workloads(model, inputs,
            output_names=("logits", "logits_ngram", "encoder_last_hidden_state"))

    def run():
        output = model(**inputs)
        cache = output.pop("past_key_values")
        return dict(output, **seq2seq_cache_outputs(cache))

    return {"forward": Workload(run=run)}
