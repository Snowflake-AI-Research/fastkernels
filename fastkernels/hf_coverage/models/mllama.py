"""Mllama constructor image/text graph with local/global vision and self/cross KV.

Learned tanh gates are fixed-weight transforms at load time. Activation gates
use ProductGate; RMSNormNative and Gemma's RoPE are unchanged internal ops.
"""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L2.gemma_dense_attention import (
    GemmaRotaryEmbedding, _apply_rotary_pos_emb,
)
from ..patches.product_gate import ProductGate
from ..runner import Workload


def multiply(op, value, gate):
    return op(torch.cat((gate.expand_as(value), value), dim=-1))


class Attention(nn.Module):
    def __init__(self, width, heads, kv_heads, norm_eps=None):
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, width // heads
        self.q_proj = Linear(width, width, bias=False)
        self.k_proj = Linear(width, kv_heads * self.dim, bias=False)
        self.v_proj = Linear(width, kv_heads * self.dim, bias=False)
        self.o_proj = Linear(width, width, bias=False)
        self.attention = DenseAttention(backend='sdpa')
        if norm_eps is not None:
            self.q_norm = RMSNormNative(self.dim, norm_eps)
            self.k_norm = RMSNormNative(self.dim, norm_eps)

    def forward(self, hidden, mask, source=None, past=None, positions=None):
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.heads, self.dim)
        cross = hasattr(self, 'q_norm')
        if cross:
            query = self.q_norm(query)
        if cross and source is None:
            key, value = past
        else:
            source = hidden if source is None else source
            key = self.k_proj(source).reshape(batch, -1, self.kv_heads, self.dim)
            value = self.v_proj(source).reshape(batch, -1, self.kv_heads, self.dim)
            if cross:
                key = self.k_norm(key)
            if positions is not None:
                query, key = _apply_rotary_pos_emb(query, key, *positions)
            if past is not None:
                key, value = torch.cat((past[0], key), 1), torch.cat((past[1], value), 1)
        cache = (key, value)
        groups = self.heads // self.kv_heads
        if groups != 1:
            key = key[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(batch, -1, self.heads, self.dim)
            value = value[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(batch, -1, self.heads, self.dim)
        output = self.attention(query, key, value, attn_mask=mask, softmax_scale=self.dim**-0.5)
        return self.o_proj(output.reshape(batch, length, -1)), cache


class MLP(nn.Module):
    def __init__(self, config, vision=False):
        super().__init__()
        self.vision = vision
        if vision:
            self.fc1 = Linear(config.hidden_size, config.intermediate_size)
            self.fc2 = Linear(config.intermediate_size, config.hidden_size)
            self.activation = GELU()
        else:
            self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
            self.activation = SiLU()
            self.product = ProductGate()

    def forward(self, hidden):
        if self.vision:
            return self.fc2(self.activation(self.fc1(hidden)))
        return self.down_proj(multiply(self.product, self.up_proj(hidden), self.activation(self.gate_proj(hidden))))


class VisionLayer(nn.Module):
    def __init__(self, config, gated):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.attention_heads, config.attention_heads)
        self.input_layernorm = LayerNorm(config.hidden_size, config.norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, config.norm_eps, promote_fp32=False)
        self.mlp = MLP(config, vision=True)
        self.gated = gated
        if gated:
            self.register_buffer('gate_attn', torch.zeros(1))
            self.register_buffer('gate_ffn', torch.zeros(1))
            self.product = ProductGate()

    def forward(self, hidden, mask):
        update, _ = self.self_attn(self.input_layernorm(hidden), mask)
        hidden = hidden + (multiply(self.product, update, self.gate_attn) if self.gated else update)
        update = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + (multiply(self.product, update, self.gate_ffn) if self.gated else update)


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, tiles = config.hidden_size, config.max_num_tiles
        self.patches = (config.image_size // config.patch_size)**2 + 1
        self.patch_embedding = Conv2d(config.num_channels, width, config.patch_size, stride=config.patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(width))
        # Fixed learned embeddings are gated during loading, retaining native
        # dtype rounding and the two separate position additions.
        self.pre_tile = Embedding(len(config.supported_aspect_ratios) + 1, tiles * width)
        self.post_tile = Embedding(len(config.supported_aspect_ratios) + 1, tiles * width)
        self.tile_position = Embedding(len(config.supported_aspect_ratios) + 1, tiles * self.patches * width)
        self.position = nn.Parameter(torch.empty(self.patches, width))
        self.layernorm_pre = LayerNorm(width, promote_fp32=False)
        self.layernorm_post = LayerNorm(width, promote_fp32=False)
        self.transformer = nn.ModuleList([VisionLayer(config, False) for _ in range(config.num_hidden_layers)])
        self.global_transformer = nn.ModuleList([VisionLayer(config, True) for _ in range(config.num_global_layers)])

    def forward(self, pixels, aspect_ratio_ids, aspect_ratio_mask):
        batch, images, tiles, channels, height, width = pixels.shape
        dim, patches = self.config.hidden_size, self.patches
        flat_batch = batch * images
        ids = aspect_ratio_ids.reshape(flat_batch, -1)
        hidden = self.patch_embedding(pixels.reshape(-1, channels, height, width)).flatten(2).transpose(1, 2)
        hidden = hidden.reshape(flat_batch, tiles, patches - 1, dim) + self.pre_tile(ids).reshape(flat_batch, tiles, 1, dim)
        hidden = hidden.reshape(-1, patches - 1, dim)
        hidden = torch.cat((self.class_embedding.expand(hidden.shape[0], 1, dim), hidden), 1)
        hidden = hidden.reshape(flat_batch, tiles, patches, dim) + self.position
        hidden = hidden + self.tile_position(ids).reshape(flat_batch, tiles, patches, dim)
        hidden = self.layernorm_pre(hidden)
        padding = (8 - patches % 8) % 8
        if padding:
            hidden = torch.cat((hidden, hidden.new_zeros(flat_batch, tiles, padding, dim)), 2)
        padded = patches + padding
        # All operations below consume supplied masks or position metadata.
        valid = aspect_ratio_mask.reshape(flat_batch, tiles, 1).bool().expand(-1, -1, padded).clone()
        valid[:, :, patches:] = False
        invalid = ~valid.reshape(flat_batch, -1)
        mask = hidden.new_zeros(flat_batch, 1, tiles * padded, tiles * padded)
        mask.masked_fill_(invalid[:, None, :, None] & invalid[:, None, None, :], torch.finfo(hidden.dtype).min)
        hidden = hidden.reshape(flat_batch, tiles * padded, dim)
        intermediate = []
        for layer in self.transformer:
            hidden = layer(hidden, mask)
            intermediate.append(hidden)
        hidden = self.layernorm_post(hidden).reshape(flat_batch, tiles, padded, dim)
        hidden = hidden + self.post_tile(ids).reshape(flat_batch, tiles, 1, dim)
        hidden = hidden.reshape(flat_batch, tiles * padded, dim)
        for layer in self.global_transformer:
            hidden = layer(hidden, mask)
        hidden = hidden.reshape(batch, images, tiles, padded, dim)[:, :, :, :patches]
        selected = torch.stack([intermediate[i] for i in self.config.intermediate_layers_indices], -1)
        selected = selected.reshape(batch, images, tiles, padded, -1)[:, :, :, :patches]
        return torch.cat((hidden, selected), -1)


class TextLayer(nn.Module):
    def __init__(self, config, cross):
        super().__init__()
        self.cross = cross
        attention = Attention(config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                              config.rms_norm_eps if cross else None)
        if cross:
            self.cross_attn = attention
            self.register_buffer('cross_attn_attn_gate', torch.zeros(1))
            self.register_buffer('cross_attn_mlp_gate', torch.zeros(1))
            self.product = ProductGate()
        else:
            self.self_attn = attention
        self.input_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormNative(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, hidden, mask, positions, image=None, past=None, row_mask=None):
        attention = self.cross_attn if self.cross else self.self_attn
        update, cache = attention(self.input_layernorm(hidden), mask, source=image if self.cross else None,
                                  past=past, positions=None if self.cross else positions)
        hidden = hidden + (multiply(self.product, update, self.cross_attn_attn_gate) if self.cross else update)
        update = self.mlp(self.post_attention_layernorm(hidden))
        if self.cross:
            if row_mask is not None:
                update = multiply(self.product, update, row_mask)
            update = multiply(self.product, update, self.cross_attn_mlp_gate)
        return hidden + update, cache


class Mllama(nn.Module):
    def __init__(self, config):
        super().__init__()
        text = config.text_config
        self.vision = Vision(config.vision_config)
        self.multi_modal_projector = Linear(config.vision_config.vision_output_dim, text.hidden_size)
        self.embed_tokens = Embedding(text.vocab_size + 8, text.hidden_size, padding_idx=text.pad_token_id)
        self.layers = nn.ModuleList([TextLayer(text, i in text.cross_attention_layers) for i in range(text.num_hidden_layers)])
        self.norm = RMSNormNative(text.hidden_size, text.rms_norm_eps)
        self.lm_head = Linear(text.hidden_size, text.vocab_size, bias=False)
        self.rotary = GemmaRotaryEmbedding(text.hidden_size // text.num_attention_heads,
                                           text.max_position_embeddings, text.rope_parameters['rope_theta'])

    def forward(self, input_ids, pixel_values=None, aspect_ratio_ids=None, aspect_ratio_mask=None,
                cross_attention_mask=None, attention_mask=None, past_key_values=None):
        hidden = self.embed_tokens(input_ids)
        image = None
        if pixel_values is not None:
            image = self.multi_modal_projector(self.vision(pixel_values, aspect_ratio_ids, aspect_ratio_mask))
        previous = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None] + previous
        rotary = self.rotary(hidden, positions)
        allowed = torch.arange(previous + hidden.shape[1], device=hidden.device)[None, None] <= positions[:, :, None]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None].bool()
        mask = hidden.new_zeros(allowed.shape).masked_fill_(~allowed, torch.finfo(hidden.dtype).min)[:, None]
        cross_mask, row_mask = None, None
        if cross_attention_mask is not None:
            supplied = cross_attention_mask[:, previous:previous + hidden.shape[1]]
            allowed_cross = supplied.repeat_interleave(self.vision.patches, dim=-1).flatten(2).bool()
            row_mask = allowed_cross.any(-1, keepdim=True)
            # HF clears all-invalid mask rows, but gates only the cross MLP.
            cross_mask = hidden.new_zeros(allowed_cross.shape).masked_fill_(~allowed_cross & row_mask, torch.finfo(hidden.dtype).min)[:, None]
            row_mask = row_mask.to(hidden.dtype)
        caches = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden, cache = layer(hidden, cross_mask if layer.cross else mask, rotary, image, past, row_mask)
            caches.append(cache)
        return {'logits': self.lm_head(self.norm(hidden)), 'past_key_values': tuple(caches)}


def build_from_config(config, device, dtype):
    text, vision = config.text_config, config.vision_config
    if (text.hidden_act != 'silu' or vision.hidden_act != 'gelu' or text.rope_parameters['rope_type'] != 'default'
            or not text.use_cache or text.tie_word_embeddings or 0 in text.cross_attention_layers
            or vision.vision_output_dim != vision.hidden_size * (1 + len(vision.intermediate_layers_indices))):
        raise ValueError('Mllama constructor adapter requires native SiLU/GELU, default RoPE and self/cross cache graph')
    model = Mllama(config).to(device=device, dtype=dtype).eval()
    with torch.device('cpu'):
        model.rotary = GemmaRotaryEmbedding(text.hidden_size // text.num_attention_heads,
                                            text.max_position_embeddings, text.rope_parameters['rope_theta']).to(device)
    return model


def load_state_dict_into(model, state_dict, config):
    mapped, used = {}, set()
    special = {
        'vision.pre_tile.emb.weight': ('pre_tile_positional_embedding.embedding.weight', 'pre_tile_positional_embedding.gate'),
        'vision.post_tile.emb.weight': ('post_tile_positional_embedding.embedding.weight', 'post_tile_positional_embedding.gate'),
        'vision.tile_position.emb.weight': ('gated_positional_embedding.tile_embedding.weight', 'gated_positional_embedding.gate'),
        'vision.position': ('gated_positional_embedding.embedding', 'gated_positional_embedding.gate'),
    }
    for name, target in model.state_dict().items():
        if name in special:
            weight_name, gate_name = ['model.vision_model.' + key for key in special[name]]
            gate = state_dict[gate_name].tanh()
            if name == 'vision.position':
                gate = 1 - gate
            value = state_dict[weight_name] * gate
            used.update((weight_name, gate_name))
        else:
            if name.startswith('vision.'):
                source = name.replace('vision.', 'model.vision_model.', 1)
                source = source.replace('.transformer.', '.transformer.layers.').replace('.global_transformer.', '.global_transformer.layers.')
            elif name.startswith(('layers.', 'norm.', 'embed_tokens.')):
                source = 'model.language_model.' + name.replace('.emb.', '.')
            elif name.startswith('multi_modal_projector.'):
                source = 'model.' + name
            else:
                source = name
            value = state_dict[source]
            used.add(source)
            if name.endswith(('gate_attn', 'gate_ffn', 'cross_attn_attn_gate', 'cross_attn_mlp_gate')):
                value = value.tanh()
        if value.shape != target.shape:
            raise ValueError(f'Mllama weight shape mismatch: {name}: {value.shape} != {target.shape}')
        mapped[name] = value
    if used != set(state_dict):
        raise KeyError(f'Unmapped Mllama state: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)


def flatten(output):
    result = {'logits': output['logits']}
    for index, (key, value) in enumerate(output['past_key_values']):
        result[f'past_key_values.{index}.key'] = key.transpose(1, 2)
        result[f'past_key_values.{index}.value'] = value.transpose(1, 2)
    return result


def make_workloads(model, inputs, config, case=None):
    if case is None or case.get('workload') != 'causal_lm_continuation':
        return {'forward': Workload(run=lambda: flatten(model(**inputs)))}
    ids = inputs['input_ids']
    prefix = ids.shape[1] - 2
    state = {}

    def initial():
        values = dict(inputs, input_ids=ids[:, :prefix])
        if 'attention_mask' in values:
            values['attention_mask'] = values['attention_mask'][:, :prefix]
        return model(**values)

    def advance(index, previous):
        end = prefix + index + 1
        return model(ids[:, end - 1:end], past_key_values=previous,
                     cross_attention_mask=inputs.get('cross_attention_mask'),
                     attention_mask=None if 'attention_mask' not in inputs else inputs['attention_mask'][:, :end])

    def prepare(index):
        previous = initial()['past_key_values']
        for step in range(index):
            previous = advance(step, previous)['past_key_values']
        state['previous'] = previous

    return {
        'prefill': Workload(run=lambda: flatten(initial())),
        'decode_1': Workload(run=lambda: flatten(advance(0, state['previous'])), prepare=lambda: prepare(0)),
        'decode_2': Workload(run=lambda: flatten(advance(1, state['previous'])), prepare=lambda: prepare(1)),
    }
