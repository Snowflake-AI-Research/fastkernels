"""T5Gemma conditional generation for the selected native SDPA computation.

SDPA ignores HF's attention-score softcap; the executed final logit cap is kept.
Gemma3's unchanged native norm, gated GELU and rotary callables preserve HF's
cast order. There is no separate optimized interface for these selected calls.
"""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.hf_coverage.models.gemma3 import MLP, NativeNorm, PositionRotary
from fastkernels.hf_coverage.runner import seq2seq_continuation_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh


class Attention(nn.Module):
    def __init__(self, config, kind, *, cross=False):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        self.scale = config.query_pre_attn_scalar ** -0.5
        self.cross = cross
        self.window = config.sliding_window if kind == 'sliding_attention' and not cross else None
        memory_width = config.cross_attention_hidden_size if cross else config.hidden_size
        self.q_proj = Linear(config.hidden_size, self.heads * self.dim, config.attention_bias)
        self.k_proj = Linear(memory_width, self.kv_heads * self.dim, config.attention_bias)
        self.v_proj = Linear(memory_width, self.kv_heads * self.dim, config.attention_bias)
        self.o_proj = Linear(self.heads * self.dim, config.hidden_size, config.attention_bias)
        if not cross:
            rotary_config = SimpleNamespace(head_dim=self.dim, rope_parameters={kind: config.rope_parameters})
            self.rotary = PositionRotary(rotary_config, kind)
        self.core = DenseAttention(backend='sdpa')

    def forward(self, hidden, positions, mask, previous=None, memory=None):
        query = self.q_proj(hidden).reshape(*hidden.shape[:2], self.heads, self.dim)
        if self.cross and previous is not None:
            key, value = previous
        else:
            source = memory if self.cross else hidden
            key = self.k_proj(source).reshape(*source.shape[:2], self.kv_heads, self.dim)
            value = self.v_proj(source).reshape(*source.shape[:2], self.kv_heads, self.dim)
            if not self.cross:
                query, key = self.rotary(query, key, positions)
            key, value = key.transpose(1, 2), value.transpose(1, 2)
            if previous is not None:
                key, value = (torch.cat((old, new), dim=2) for old, new in zip(previous, (key, value)))
            else:
                key, value = (x.clone(memory_format=torch.contiguous_format) for x in (key, value))
        retained = (key, value)
        if self.window is not None:
            retained = (key[:, :, -self.window + 1:], value[:, :, -self.window + 1:])
        groups = self.heads // self.kv_heads
        key, value = (x.transpose(1, 2).repeat_interleave(groups, dim=2) for x in (key, value))
        output = self.core(query, key, value, softmax_scale=self.scale, attn_mask=mask)
        return self.o_proj(output.reshape(*hidden.shape[:2], -1)), retained


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.self_attn = Attention(config, kind)
        self.mlp = MLP(config)
        branches = ['self_attn', 'feedforward']
        if config.is_decoder:
            self.cross_attn = Attention(config, kind, cross=True)
            branches.append('cross_attn')
        for branch in branches:
            for side in ('pre', 'post'):
                setattr(self, f'{side}_{branch}_layernorm', NativeNorm(config.hidden_size, config.rms_norm_eps))

    def forward(self, hidden, positions, mask, previous=None, memory=None, cross_mask=None):
        branch, self_cache = self.self_attn(
            self.pre_self_attn_layernorm(hidden), positions, mask,
            None if previous is None else previous[0])
        hidden = hidden + self.post_self_attn_layernorm(branch)
        cross_cache = None
        if memory is not None:
            branch, cross_cache = self.cross_attn(
                self.pre_cross_attn_layernorm(hidden), positions, cross_mask,
                None if previous is None else previous[1], memory)
            hidden = hidden + self.post_cross_attn_layernorm(branch)
        hidden = hidden + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))
        return hidden, (self_cache, cross_cache)


class Stack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(Layer(config, kind) for kind in config.layer_types)
        self.norm = NativeNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, ids, attention_mask=None, memory=None, cross_mask=None, previous=None):
        config = self.config
        seen = 0
        if previous is not None:
            # The selected alternating stack contains full-attention layers.
            full_index = config.layer_types.index('full_attention')
            seen = previous[full_index][0][0].shape[2]
        positions = torch.arange(seen, seen + ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        hidden = self.embed_tokens(ids)
        hidden = hidden * torch.tensor(config.hidden_size ** 0.5, device=hidden.device, dtype=hidden.dtype)
        if attention_mask is None and not config.is_decoder:
            attention_mask = ids != config.pad_token_id
        masks = {}
        for kind in set(config.layer_types):
            offset = max(seen - config.sliding_window + 1, 0) if config.is_decoder and kind == 'sliding_attention' else 0
            keys = torch.arange(offset, seen + ids.shape[1], device=ids.device)
            delta = positions[0, :, None] - keys[None, :]
            allowed = delta >= 0 if config.is_decoder else torch.ones_like(delta, dtype=torch.bool)
            if kind == 'sliding_attention':
                allowed = allowed & (delta < config.sliding_window if config.is_decoder else delta.abs() <= config.sliding_window)
            allowed = allowed[None, None].expand(ids.shape[0], 1, -1, -1)
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, offset:].bool()
            masks[kind] = allowed
        caches = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, positions, masks[config.layer_types[index]],
                                  None if previous is None else previous[index], memory, cross_mask)
            caches.append(cache)
        return self.norm(hidden), tuple(caches)


class T5Gemma(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.encoder = Stack(config.encoder)
        self.model.decoder = Stack(config.decoder)
        self.lm_head = nn.Module()
        self.lm_head.out_proj = Linear(config.decoder.hidden_size, config.decoder.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.out_proj.weight = self.model.decoder.embed_tokens.emb.weight
        self.cap = Tanh()

    def forward(self, input_ids, decoder_input_ids, *, encoder_hidden_states=None,
                past_key_values=None, attention_mask=None, decoder_attention_mask=None):
        memory = encoder_hidden_states
        if memory is None:
            memory, _ = self.model.encoder(input_ids, attention_mask)
        cross_mask = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        hidden, cache = self.model.decoder(decoder_input_ids, decoder_attention_mask, memory, cross_mask, past_key_values)
        logits = self.lm_head.out_proj(hidden)
        softcap = self.config.decoder.final_logit_softcapping
        if softcap is not None:
            logits = self.cap(logits / softcap) * softcap
        return {'logits': logits, 'encoder_last_hidden_state': memory,
                'decoder_hidden_states.0': hidden, 'past_key_values': cache}


def build_from_config(config, device, dtype):
    if _tp_size() != 1 or not config.is_encoder_decoder:
        raise ValueError('T5Gemma coverage requires single-rank encoder-decoder inference')
    for part in (config.encoder, config.decoder):
        if (part.hidden_activation != 'gelu_pytorch_tanh' or part.attention_bias
                or part.rope_parameters['rope_type'] != 'default'
                or set(part.layer_types) != {'sliding_attention', 'full_attention'}
                or part.sliding_window < 2):
            raise ValueError('T5Gemma coverage preserves alternating attention, default RoPE and bias-free tanh-GeGLU')
    return T5Gemma(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    if config.tie_word_embeddings and not torch.equal(
            state_dict['lm_head.out_proj.weight'], state_dict['model.decoder.embed_tokens.weight']):
        raise ValueError('T5Gemma requires tied decoder embeddings and output head')
    expected = model.state_dict()
    mapped, consumed = {}, set()
    for name in expected:
        source = name.replace('.embed_tokens.emb.', '.embed_tokens.')
        mapped[name] = state_dict[source]
        consumed.add(source)
    if consumed != set(state_dict):
        raise ValueError(f'Unmapped T5Gemma state: {sorted(set(state_dict) - consumed)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return seq2seq_continuation_workloads(
        model, inputs, output_names=('logits', 'encoder_last_hidden_state', 'decoder_hidden_states.0'))
