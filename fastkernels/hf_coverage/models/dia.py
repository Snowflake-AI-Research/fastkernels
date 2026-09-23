"""Dia's text encoder and nine-channel audio-token decoder with explicit caches."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload, seq2seq_cache_outputs
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.moe_sum import MoeSum
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul
from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm


class AudioMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = SiluAndMul()

    def forward(self, hidden):
        return self.down_proj(self.activation(self.gate_up_proj(hidden)))


class AudioAttention(nn.Module):
    def __init__(self, config, cross=False):
        super().__init__()
        self.cross = cross
        self.heads = config.cross_num_attention_heads if cross else config.num_attention_heads
        self.kv_heads = config.cross_num_key_value_heads if cross else config.num_key_value_heads
        self.width = config.cross_head_dim if cross else config.head_dim
        source_width = config.cross_hidden_size if cross else config.hidden_size
        self.q_proj = Linear(config.hidden_size, self.heads * self.width, bias=False)
        self.k_proj = Linear(source_width, self.kv_heads * self.width, bias=False)
        self.v_proj = Linear(source_width, self.kv_heads * self.width, bias=False)
        self.o_proj = Linear(self.heads * self.width, config.hidden_size, bias=False)
        self.attention = DenseAttention(backend="sdpa")

    def forward(self, hidden, memory=None, rotary=None, positions=None, mask=None, cache=None, causal=False):
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).reshape(batch, length, self.heads, self.width)
        if self.cross and cache is not None:
            key, value = cache
        else:
            source = hidden if memory is None else memory
            key, value = (projection(source).reshape(batch, -1, self.kv_heads, self.width)
                          for projection in (self.k_proj, self.v_proj))
            if rotary is not None:
                query, key = rotary(positions.reshape(-1), query.reshape(batch * length, -1),
                                    key.reshape(batch * length, -1))
                query = query.reshape(batch, length, self.heads, self.width)
                key = key.reshape(batch, length, self.kv_heads, self.width)
            key, value = key.transpose(1, 2), value.transpose(1, 2)
            if cache is not None:
                key, value = (torch.cat((old, new), dim=2) for old, new in zip(cache, (key, value)))
            else:
                key, value = key.contiguous(), value.contiguous()
        source_length = key.shape[2]
        attention_mask = None if mask is None else mask[:, None, None, :]
        if causal:
            allowed = torch.arange(source_length, device=hidden.device)[None, :] <= positions[0, :, None]
            attention_mask = allowed if attention_mask is None else attention_mask & allowed
        # GQA repetition is only a layout/copy, preserving HF's SDPA grouping.
        groups = self.heads // self.kv_heads
        k, v = (tensor[:, :, None].expand(-1, -1, groups, -1, -1)
                .reshape(batch, self.heads, source_length, self.width).transpose(1, 2)
                for tensor in (key, value))
        output = self.attention(query, k, v, attn_mask=attention_mask, softmax_scale=1.0)
        return self.o_proj(output.reshape(batch, length, -1)), (key, value)


class AudioLayer(nn.Module):
    def __init__(self, config, decoder=False):
        super().__init__()
        self.decoder = decoder
        self.pre_sa_norm = T5LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attention = AudioAttention(config)
        self.mlp = AudioMLP(config)
        if decoder:
            self.pre_ca_norm = T5LayerNorm(config.hidden_size, eps=config.norm_eps)
            self.cross_attention = AudioAttention(config, cross=True)
            self.pre_mlp_norm = T5LayerNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.post_sa_norm = T5LayerNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden, rotary, positions, mask=None, memory=None, cache=None):
        attention, self_cache = self.self_attention(self.pre_sa_norm(hidden), rotary=rotary, positions=positions,
                                                    mask=None if self.decoder else mask, causal=self.decoder,
                                                    cache=None if cache is None else cache[0])
        hidden = hidden + attention
        cross_cache = None
        if self.decoder:
            attention, cross_cache = self.cross_attention(self.pre_ca_norm(hidden), memory=memory, mask=mask,
                                                          cache=None if cache is None else cache[1])
            hidden = hidden + attention
        norm = self.pre_mlp_norm if self.decoder else self.post_sa_norm
        return hidden + self.mlp(norm(hidden)), (self_cache, cross_cache)


class Dia(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for name, tower, decoder in (("encoder", config.encoder_config, False),
                                     ("decoder", config.decoder_config, True)):
            stack = nn.Module()
            stack.embedding = Embedding(tower.vocab_size * (tower.num_channels if decoder else 1), tower.hidden_size)
            stack.layers = nn.ModuleList(AudioLayer(tower, decoder) for _ in range(tower.num_hidden_layers))
            stack.norm = T5LayerNorm(tower.hidden_size, eps=tower.norm_eps)
            stack.rotary = RotaryEmbedding(tower.head_dim, tower.max_position_embeddings, tower.rope_parameters["rope_theta"])
            setattr(self, name, stack)
        self.logits_dense = Linear(config.decoder_config.hidden_size,
                                   config.decoder_config.num_channels * config.decoder_config.vocab_size, bias=False)
        self.channel_sum = MoeSum()

    def encode(self, ids, mask=None):
        hidden = self.encoder.embedding(ids)
        positions = torch.arange(ids.shape[1], device=ids.device).expand(ids.shape[0], -1)
        for layer in self.encoder.layers:
            hidden, _ = layer(hidden, self.encoder.rotary, positions, mask=mask)
        return self.encoder.norm(hidden)

    def decode(self, ids, memory, mask=None, cache=None):
        config = self.config.decoder_config
        offsets = torch.arange(config.num_channels, device=ids.device) * config.vocab_size
        embedded = self.decoder.embedding(ids + offsets)
        # The existing expert-output reduction supports this identical channel
        # axis; nine channels select its native general reduction implementation.
        hidden = self.channel_sum(embedded.reshape(-1, config.hidden_size), config.num_channels)
        hidden = hidden.reshape(*ids.shape[:2], config.hidden_size)
        offset = 0 if cache is None else cache[0][0][0].shape[2]
        positions = (torch.arange(ids.shape[1], device=ids.device) + offset).expand(ids.shape[0], -1)
        next_cache = []
        for index, layer in enumerate(self.decoder.layers):
            hidden, state = layer(hidden, self.decoder.rotary, positions, mask=mask, memory=memory,
                                  cache=None if cache is None else cache[index])
            next_cache.append(state)
        hidden = self.decoder.norm(hidden)
        logits = self.logits_dense(hidden).reshape(*ids.shape[:2], config.num_channels, config.vocab_size)
        logits = logits.transpose(1, 2).reshape(ids.shape[0] * config.num_channels, ids.shape[1], config.vocab_size)
        return {"logits": logits, "encoder_last_hidden_state": memory,
                **seq2seq_cache_outputs(next_cache)}, tuple(next_cache)

    def forward(self, input_ids, decoder_input_ids, attention_mask=None):
        mask = None if attention_mask is None else attention_mask.bool()
        memory = self.encode(input_ids, mask)
        return self.decode(decoder_input_ids, memory, mask)[0]


def build_from_config(config, device, dtype):
    for tower in (config.encoder_config, config.decoder_config):
        if tower.hidden_act != "silu" or tower.rope_parameters["rope_type"] != "default":
            raise ValueError("Dia's published path requires SiLU and ordinary RoPE")
    return Dia(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for key in model.state_dict():
        if key == "logits_dense.weight":
            source = key
        else:
            source = "model." + key.replace(".embedding.emb.", ".embedding.")
            source = source.replace("model.decoder.embedding.", "model.decoder.embeddings.embed.")
        mapped[key] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped Dia state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
