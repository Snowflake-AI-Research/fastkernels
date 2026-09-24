"""T5Gemma2 multimodal encoder and merged-attention cached decoder.

Reuses existing SigLIP, Gemma normalization/MLP/rotary/pooling, and BMM/Softmax
operations. Cross keys share decoder projections but receive no rotary embedding.
"""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L4.pi0 import SigLIPVisionEncoder
from ..runner import Workload, seq2seq_cache_outputs
from .gemma3 import NativeNorm, MLP, PositionRotary, Projector
from .siglip import configure_encoder


class VisionAttentionCore(nn.Module):
    """Existing BMM/Softmax composition preserving eager SigLIP BF16 cast order."""

    def __init__(self):
        super().__init__()
        self.matmul, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, query, key, value, softmax_scale=None, causal=False):
        if causal:
            raise ValueError('SigLIP vision attention is bidirectional')
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        scores = self.matmul(query, key.transpose(-1, -2)) * softmax_scale
        weights = self.softmax(scores.float()).to(query.dtype)
        return self.matmul(weights, value).transpose(1, 2)


class ScaledEmbedding(Embedding):
    def __init__(self, config, eoi):
        super().__init__(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.scale, self.eoi = config.hidden_size ** 0.5, eoi
        self.eoi_embedding = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, ids):
        hidden = super().forward(ids)
        hidden = hidden * torch.tensor(self.scale, device=hidden.device, dtype=hidden.dtype)
        hidden[ids == self.eoi] = self.eoi_embedding.to(hidden.dtype)
        return hidden


class Attention(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.heads, self.kv_heads, self.dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.window = config.sliding_window if kind == 'sliding_attention' else None
        self.scale = config.query_pre_attn_scalar ** -0.5
        for name, width in (('q_proj', self.heads*self.dim), ('k_proj', self.kv_heads*self.dim),
                            ('v_proj', self.kv_heads*self.dim)):
            setattr(self, name, Linear(config.hidden_size, width, config.attention_bias))
        self.o_proj = Linear(self.heads*self.dim, config.hidden_size, config.attention_bias)
        self.q_norm, self.k_norm = (NativeNorm(self.dim, config.rms_norm_eps) for _ in range(2))
        self.rotary = PositionRotary(config, kind)
        self.matmul, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, hidden, positions, mask, memory=None, previous=None):
        shape = (*hidden.shape[:2], -1, self.dim)
        query = self.q_norm(self.q_proj(hidden).reshape(shape))
        key = self.k_norm(self.k_proj(hidden).reshape(shape))
        value = self.v_proj(hidden).reshape(shape).transpose(1, 2)
        query, key = self.rotary(query, key, positions)
        query, key = query.transpose(1, 2), key.transpose(1, 2)
        if previous is not None:
            key, value = (torch.cat((old, new), dim=2) for old, new in zip(previous[0], (key, value)))
        retained = None
        if memory is not None:
            self_cache = ((key[:, :, -self.window+1:], value[:, :, -self.window+1:])
                          if self.window else (key, value))
            if previous is None:
                cross_shape = (*memory.shape[:2], -1, self.dim)
                cross_key = self.k_norm(self.k_proj(memory).reshape(cross_shape)).transpose(1, 2)
                cross_value = self.v_proj(memory).reshape(cross_shape).transpose(1, 2)
            else:
                cross_key, cross_value = previous[1]
            retained = (self_cache, (cross_key, cross_value))
            key, value = torch.cat((key, cross_key), 2), torch.cat((value, cross_value), 2)
        groups = self.heads // self.kv_heads
        key, value = (x.repeat_interleave(groups, dim=1) for x in (key, value))
        scores = self.matmul(query, key.transpose(-1, -2)) * self.scale + mask
        weights = self.softmax(scores.float()).to(query.dtype)
        result = self.matmul(weights, value).transpose(1, 2).contiguous().reshape(*hidden.shape[:2], -1)
        return self.o_proj(result), retained


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.self_attn, self.mlp = Attention(config, kind), MLP(config)
        for name in ('pre_self_attn_layernorm', 'post_self_attn_layernorm',
                     'pre_feedforward_layernorm', 'post_feedforward_layernorm'):
            setattr(self, name, NativeNorm(config.hidden_size, config.rms_norm_eps))

    def forward(self, hidden, positions, mask, memory=None, previous=None):
        attention, cache = self.self_attn(self.pre_self_attn_layernorm(hidden), positions, mask, memory, previous)
        hidden = hidden + self.post_self_attn_layernorm(attention)
        hidden = hidden + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden)))
        return hidden, cache


def text_stack(config, eoi):
    stack = nn.Module()
    stack.embed_tokens = ScaledEmbedding(config, eoi)
    stack.layers = nn.ModuleList(Layer(config, kind) for kind in config.layer_types)
    stack.norm = NativeNorm(config.hidden_size, config.rms_norm_eps)
    return stack


def additive_mask(allowed, hidden):
    return torch.zeros(allowed.shape, device=hidden.device, dtype=hidden.dtype).masked_fill(
        ~allowed, torch.finfo(hidden.dtype).min)


class T5Gemma2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        encoder = self.model.encoder = nn.Module()
        encoder.text_model = text_stack(config.encoder.text_config, config.eoi_token_index)
        encoder.vision_tower = SigLIPVisionEncoder(config.encoder.vision_config)
        configure_encoder(encoder.vision_tower.layers)
        for layer in encoder.vision_tower.layers:
            layer.self_attn.attn = VisionAttentionCore()
        encoder.vision_tower.post_layernorm.promote_fp32 = False
        encoder.multi_modal_projector = Projector(config.encoder)
        self.model.decoder = text_stack(config.decoder, config.eoi_token_index)
        self.model.decoder.embed_tokens.emb.weight = encoder.text_model.embed_tokens.emb.weight
        self.model.decoder.embed_tokens.eoi_embedding = encoder.text_model.embed_tokens.eoi_embedding
        self.lm_head = nn.Module()
        self.lm_head.out_proj = Linear(config.decoder.hidden_size, config.decoder.vocab_size, bias=False)
        self.lm_head.out_proj.weight = encoder.text_model.embed_tokens.emb.weight

    def encode(self, ids, pixel_values, attention_mask):
        encoder, config = self.model.encoder, self.config.encoder.text_config
        hidden = encoder.text_model.embed_tokens(ids)
        if pixel_values is not None:
            image = encoder.multi_modal_projector(encoder.vision_tower(pixel_values)).to(hidden.dtype)
            mask = (ids == self.config.image_token_index)[..., None].expand_as(hidden)
            hidden = hidden.masked_scatter(mask, image)
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        distance = positions[0, :, None] - positions[0, None, :]
        masks = {}
        for kind in set(config.layer_types):
            allowed = torch.ones_like(distance, dtype=torch.bool)
            if kind == 'sliding_attention':
                allowed = (distance < (config.sliding_window+1)//2) & (distance > -(config.sliding_window//2+1))
            allowed = allowed[None, None].expand(ids.shape[0], 1, -1, -1)
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].bool()
            masks[kind] = additive_mask(allowed, hidden)
        for layer, kind in zip(encoder.text_model.layers, config.layer_types):
            hidden, _ = layer(hidden, positions, masks[kind])
        return encoder.text_model.norm(hidden)

    def forward(self, input_ids, decoder_input_ids, pixel_values=None, attention_mask=None,
                decoder_attention_mask=None, encoder_hidden_states=None, past_key_values=None, seen=0):
        memory = encoder_hidden_states
        if memory is None:
            memory = self.encode(input_ids, pixel_values, attention_mask)
        hidden = self.model.decoder.embed_tokens(decoder_input_ids)
        config = self.config.decoder
        positions = torch.arange(seen, seen+decoder_input_ids.shape[1], device=hidden.device)
        masks = {}
        for kind in set(config.layer_types):
            offset = max(seen-config.sliding_window+1, 0) if kind == 'sliding_attention' else 0
            keys = torch.arange(offset, seen+decoder_input_ids.shape[1], device=hidden.device)
            allowed = keys[None, :] <= positions[:, None]
            if kind == 'sliding_attention':
                allowed = allowed & (positions[:, None]-keys[None, :] < config.sliding_window)
            allowed = allowed[None, None].expand(hidden.shape[0], 1, -1, -1)
            if decoder_attention_mask is not None:
                allowed = allowed & decoder_attention_mask[:, None, None, offset:].bool()
            cross = torch.ones((*allowed.shape[:-1], memory.shape[1]), device=hidden.device, dtype=torch.bool)
            if attention_mask is not None:
                cross = cross & attention_mask[:, None, None, :].bool()
            masks[kind] = additive_mask(torch.cat((allowed, cross), -1), hidden)
        positions = positions[None].expand(hidden.shape[0], -1)
        cache = []
        for index, (layer, kind) in enumerate(zip(self.model.decoder.layers, config.layer_types)):
            hidden, state = layer(hidden, positions, masks[kind], memory,
                                  None if past_key_values is None else past_key_values[index])
            cache.append(state)
        return {'logits': self.lm_head.out_proj(self.model.decoder.norm(hidden)),
                'encoder_last_hidden_state': memory, 'past_key_values': cache,
                'seen': seen+decoder_input_ids.shape[1]}


def build_from_config(config, device, dtype):
    text, decoder, vision = config.encoder.text_config, config.decoder, config.encoder.vision_config
    if (not config.tie_word_embeddings or text.hidden_size != decoder.hidden_size
            or text.vocab_size != decoder.vocab_size or vision.vision_use_head
            or vision.hidden_act != 'gelu_pytorch_tanh' or vision.num_channels != 3):
        raise ValueError('T5Gemma2 requires its selected tied multimodal configuration')
    for tower in (text, decoder):
        if (tower.hidden_activation != 'gelu_pytorch_tanh' or tower.attn_logit_softcapping is not None
                or tower.final_logit_softcapping is not None or not tower.use_cache
                or tower.use_bidirectional_attention):
            raise ValueError('T5Gemma2 requires the selected uncapped GELU attention blocks')
    return T5Gemma2(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    expected, mapped, used = model.state_dict(), {}, set()
    for name in expected:
        source = name.replace('.embed_tokens.emb.', '.embed_tokens.')
        prefix = 'model.encoder.vision_tower.'
        if name.startswith(prefix):
            suffix = name[len(prefix):].replace('layers.', 'encoder.layers.')
            if suffix.startswith('patch_embedding.'):
                suffix = 'embeddings.' + suffix
            elif suffix == 'position_embedding':
                suffix = 'embeddings.position_embedding.weight'
            source = prefix + suffix
        mapped[name] = state_dict[source].reshape(expected[name].shape)
        used.add(source)
    for suffix in ('weight', 'eoi_embedding'):
        if not torch.equal(state_dict['model.encoder.text_model.embed_tokens.'+suffix],
                           state_dict['model.decoder.embed_tokens.'+suffix]):
            raise ValueError('T5Gemma2 encoder and decoder embeddings must be tied')
    if not torch.equal(state_dict['lm_head.out_proj.weight'], state_dict['model.encoder.text_model.embed_tokens.weight']):
        raise ValueError('T5Gemma2 output embedding must be tied')
    if used != set(state_dict):
        raise ValueError(f'Unmapped T5Gemma2 weights: {sorted(set(state_dict)-used)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, case=None):
    ids, state = inputs['decoder_input_ids'], {}
    prefix = ids.shape[1]-2

    def call(start, end, previous=None):
        kwargs = dict(inputs, decoder_input_ids=ids[:, start:end])
        if 'decoder_attention_mask' in kwargs:
            kwargs['decoder_attention_mask'] = kwargs['decoder_attention_mask'][:, :end]
        if previous is not None:
            kwargs.update(encoder_hidden_states=previous['encoder_last_hidden_state'],
                          past_key_values=previous['past_key_values'], seen=previous['seen'])
        return model(**kwargs)

    def prepare(index):
        state['previous'] = call(0, prefix)
        for step in range(index):
            state['previous'] = call(prefix+step, prefix+step+1, state['previous'])

    def retain(output):
        state['output'] = output
        return {'logits': output['logits']}

    def collect(_):
        output = state.pop('output')
        result = {name: output[name] for name in ('logits', 'encoder_last_hidden_state')}
        result.update(seq2seq_cache_outputs(output['past_key_values']))
        return result

    return {'prefill': Workload(run=lambda: retain(call(0, prefix)), collect=collect),
            **{f'decode_{i+1}': Workload(prepare=lambda i=i: prepare(i),
                                       run=lambda i=i: retain(call(prefix+i, prefix+i+1, state['previous'])),
                                       collect=collect) for i in range(2)}}
