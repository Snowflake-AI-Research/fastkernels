"""Cohere ASR constructor defaults: Parakeet encoder and cached ReLU decoder."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.runner import seq2seq_continuation_workloads
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from .parakeet import Subsampling, RelativePositions, Block
from ..patches.audio_query_bias import BiasedQueryAttention


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.head_dim
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias))
        self.attention = DenseAttention(backend='sdpa')

    def forward(self, hidden, source, previous, mask, *, cross=False):
        batch, length = hidden.shape[:2]
        query = self.q_proj(hidden).view(batch, length, self.heads, self.width)
        if cross and previous is not None:
            key, value = previous
        else:
            key, value = (getattr(self, name)(source).view(batch, -1, self.heads, self.width)
                          .transpose(1, 2).contiguous() for name in ('k_proj', 'v_proj'))
            if previous is not None:
                key, value = (torch.cat((old, new), dim=2) for old, new in zip(previous, (key, value)))
            else:
                key, value = key.clone(), value.clone()
        context = self.attention(query, key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask,
                                 softmax_scale=self.width**-0.5)
        return self.o_proj(context.reshape(batch, length, -1)), (key, value)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.encoder_attn = Attention(config), Attention(config)
        for name in ('input_layernorm', 'post_attention_layernorm', 'final_layernorm'):
            setattr(self, name, LayerNorm(config.hidden_size, promote_fp32=False))
        self.mlp = nn.Module()
        self.mlp.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.mlp.fc2 = Linear(config.intermediate_size, config.hidden_size)
        self.activation = ReLU()

    def forward(self, hidden, memory, previous, self_mask, cross_mask):
        self_previous, cross_previous = (None, None) if previous is None else previous
        normalized = self.input_layernorm(hidden)
        update, self_cache = self.self_attn(normalized, normalized, self_previous, self_mask)
        hidden = hidden + update
        update, cross_cache = self.encoder_attn(self.post_attention_layernorm(hidden), memory,
                                               cross_previous, cross_mask, cross=True)
        hidden = hidden + update
        hidden = hidden + self.mlp.fc2(self.activation(self.mlp.fc1(self.final_layernorm(hidden))))
        return hidden, (self_cache, cross_cache)


class CohereAsr(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        enc = config.encoder_config
        self.encoder = nn.Module()
        self.encoder.subsampling = Subsampling(enc)
        self.encoder.layers = nn.ModuleList(Block(enc) for _ in range(enc.num_hidden_layers))
        for layer in self.encoder.layers:
            # Select the same native SDPA dispatch as the reference, including
            # encoder head160 (outside the cuDNN-only head128 limit).
            layer.self_attn.attention = BiasedQueryAttention(backend="sdpa")
        self.input_scale = math.sqrt(enc.hidden_size) if enc.scale_input else 1.0
        self.decoder = nn.Module()
        self.decoder.embed_tokens = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.decoder.pos_emb = Embedding(config.max_position_embeddings, config.hidden_size)
        self.decoder.embedding_layernorm = LayerNorm(config.hidden_size, promote_fp32=False)
        self.decoder.norm = LayerNorm(config.hidden_size, promote_fp32=False)
        self.decoder.proj = Linear(enc.hidden_size, config.hidden_size)
        self.decoder.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.proj_out = Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_features, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        memory = encoder_hidden_states
        if memory is None:
            memory, _ = self.encoder.subsampling(input_features, attention_mask)
            memory = memory * self.input_scale
            positions = self.encoder.encode_positions(memory)
            mask = self.encoder_mask(attention_mask, memory.shape[1])
            for layer in self.encoder.layers:
                memory = layer(memory, positions, mask)
        else:
            mask = self.encoder_mask(attention_mask, memory.shape[1])
        # Native decoder projects memory on every call even when cross KV is cached.
        projected = self.decoder.proj(memory)
        batch, length = decoder_input_ids.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0][0].shape[2]
        positions = torch.arange(length, device=decoder_input_ids.device) + past_length
        hidden = self.decoder.embedding_layernorm(
            self.decoder.embed_tokens(decoder_input_ids) + self.decoder.pos_emb(positions))
        keys = torch.arange(past_length + length, device=hidden.device)
        allowed = (keys[None, :] <= positions[:, None])[None, None].expand(batch, 1, -1, -1)
        if decoder_attention_mask is not None:
            allowed = allowed & decoder_attention_mask[:, None, None, :].bool()
        self_mask = allowed
        cross_mask = None if mask is None else mask[:, None, None, :]
        cache = []
        for index, layer in enumerate(self.decoder.layers):
            previous = None if past_key_values is None else past_key_values[index]
            hidden, updated = layer(hidden, projected, previous, self_mask, cross_mask)
            cache.append(updated)
        return {'logits': self.proj_out(self.decoder.norm(hidden)),
                'encoder_last_hidden_state': memory, 'past_key_values': tuple(cache)}

    def encoder_mask(self, attention_mask, length):
        if attention_mask is None:
            return None
        enc = self.config.encoder_config
        lengths = attention_mask.sum(-1)
        for _ in range(int(math.log2(enc.subsampling_factor))):
            lengths = (lengths - 1) // enc.subsampling_conv_stride + 1
        return torch.arange(length, device=attention_mask.device)[None] < lengths[:, None]


def build_from_config(config, device, dtype):
    enc = config.encoder_config
    if (config.hidden_act != 'relu' or enc.hidden_act != 'silu' or config.tie_word_embeddings
            or config.num_key_value_heads != config.num_attention_heads
            or config.hidden_size != config.num_attention_heads * config.head_dim
            or enc.num_key_value_heads != enc.num_attention_heads
            or enc.subsampling_conv_kernel_size % 2 != 1):
        raise ValueError('Cohere ASR requires constructor-default activation, head, and odd-kernel structure')
    model = CohereAsr(config).to(device=device, dtype=dtype)
    model.encoder.encode_positions = RelativePositions(enc, device)
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state, config):
    mapped, consumed = {}, set()
    for target, tensor in model.state_dict().items():
        source = target if target.startswith('proj_out.') else 'model.' + target
        source = source.replace('.emb.weight', '.weight')
        if source not in state or state[source].shape != tensor.shape:
            raise ValueError(f'Cohere ASR state mismatch: {target} <- {source}')
        mapped[target] = state[source]
        consumed.add(source)
    if consumed != set(state):
        raise ValueError(f'Cohere ASR unmapped state: {sorted(set(state) - consumed)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return seq2seq_continuation_workloads(model, inputs, encoder_input_name='input_features')
