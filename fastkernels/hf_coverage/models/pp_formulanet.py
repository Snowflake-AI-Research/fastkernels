"""PP-FormulaNet recognition through its native cached greedy-generation contract."""

import math
import re
import torch
from torch import nn
from .sam import VisionEncoder
from ..patches.codec_top1 import CodecTop1
from ..runner import Workload, seq2seq_cache_outputs
from fastkernels.tasks.baseline.L1.conv2d import Conv2d
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.gelu import GELU


class _Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = Conv2d(config.post_conv_in_channels, config.post_conv_mid_channels, 3, stride=2, padding=1, bias=False)
        self.conv2 = Conv2d(config.post_conv_mid_channels, config.post_conv_out_channels, 3, stride=2, padding=1, bias=False)
        self.linear_1 = Linear(config.post_conv_out_channels, config.post_conv_out_channels)
        self.linear_2 = Linear(config.post_conv_out_channels, config.decoder_hidden_size)

    def forward(self, hidden):
        hidden = self.conv2(self.conv1(hidden)).flatten(2).transpose(1, 2)
        return self.linear_2(self.linear_1(hidden))


class _Vision(VisionEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.multi_modal_projector = _Projector(config)

    def forward(self, pixels):
        hidden, _ = super().forward(pixels)
        return hidden, self.multi_modal_projector(hidden)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.heads = config.decoder_attention_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (Linear(width, width) for _ in range(4))
        self.attend = DenseAttention(backend='sdpa')

    def forward(self, hidden, memory=None, cache=None):
        batch, length, width = hidden.shape
        shape = lambda x: x.reshape(batch, -1, self.heads, width // self.heads)
        query = shape(self.q_proj(hidden))
        if memory is not None and cache is not None:
            key, value = cache
        else:
            source = hidden if memory is None else memory
            key, value = shape(self.k_proj(source)).transpose(1, 2), shape(self.v_proj(source)).transpose(1, 2)
            if cache is not None:
                key, value = torch.cat((cache[0], key), dim=2), torch.cat((cache[1], value), dim=2)
            else:
                key, value = key.clone(memory_format=torch.contiguous_format), value.clone(memory_format=torch.contiguous_format)
        attended = self.attend(query, key.transpose(1, 2), value.transpose(1, 2), causal=memory is None and length > 1)
        return self.out_proj(attended.reshape(batch, length, width)), (key, value)


class _Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.self_attn, self.encoder_attn = _Attention(config), _Attention(config)
        self.self_attn_layer_norm, self.encoder_attn_layer_norm, self.final_layer_norm = (LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(3))
        self.fc1, self.fc2, self.activation = Linear(width, config.decoder_ffn_dim), Linear(config.decoder_ffn_dim, width), GELU()

    def forward(self, hidden, memory, cache):
        attended, self_cache = self.self_attn(self.self_attn_layer_norm(hidden), cache=None if cache is None else cache[0])
        hidden = hidden + attended
        attended, cross_cache = self.encoder_attn(self.encoder_attn_layer_norm(hidden), memory, None if cache is None else cache[1])
        hidden = hidden + attended
        return hidden + self.fc2(self.activation(self.fc1(self.final_layer_norm(hidden)))), (self_cache, cross_cache)


class _Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.embed_tokens = Embedding(config.vocab_size, width, padding_idx=config.pad_token_id)
        self.embed_positions = Embedding(config.max_position_embeddings + 2, width)
        self.layernorm_embedding, self.layer_norm = (LayerNorm(width, eps=1e-5, promote_fp32=False) for _ in range(2))
        self.layers = nn.ModuleList([_Layer(config) for _ in range(config.decoder_layers)])
        self.scale = math.sqrt(width) if config.scale_embedding else 1.

    def forward(self, ids, memory, cache=None):
        offset = 0 if cache is None else cache[0][0][0].shape[2]
        positions = torch.arange(offset + 2, offset + 2 + ids.shape[1], device=ids.device)
        hidden = self.layernorm_embedding(self.embed_tokens(ids) * self.scale + self.embed_positions(positions))
        states = []
        for i, layer in enumerate(self.layers):
            hidden, state = layer(hidden, memory, None if cache is None else cache[i])
            states.append(state)
        return self.layer_norm(hidden), tuple(states)


class _Formula(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.encoder = _Vision(config.vision_config)
        self.model.decoder = _Decoder(config.text_config)
        self.lm_head = Linear(config.text_config.d_model, config.text_config.vocab_size, bias=False)
        self.top1 = CodecTop1()

    def generate(self, pixel_values, generation, steps):
        _, memory = self.model.encoder(pixel_values)
        ids = torch.full((pixel_values.shape[0], 1), generation['decoder_start_token_id'], dtype=torch.long, device=pixel_values.device)
        if ids.shape[0] != 1:
            raise ValueError('This declared recognition workload has one image')
        sequence, cache, logits = ids, None, []
        for index in range(steps):
            hidden, cache = self.model.decoder(ids, memory, cache)
            raw = self.lm_head(hidden)[:, -1].float()
            logits.append(raw)
            scores = raw
            if index + 1 == steps and generation.get('forced_eos_token_id') is not None:
                scores = torch.full_like(raw, -float('inf'))
                scores[:, generation['forced_eos_token_id']] = 0
            ids = self.top1(scores).long().reshape(1, 1)
            sequence = torch.cat((sequence, ids), dim=1)
            if ids.item() == generation.get('eos_token_id'):
                break
        return {'sequences': sequence, **{f'logits.{i}': x for i, x in enumerate(logits)}, **seq2seq_cache_outputs(cache)}


def build_from_config(config, device, dtype):
    if (not config.vision_config.use_abs_pos or not config.vision_config.use_rel_pos
            or config.text_config.activation_function != 'gelu' or not config.text_config.use_cache
            or config.text_config.tie_word_embeddings):
        raise ValueError('Preserve native SAM vision, untied pre-norm GELU decoder and caching')
    return _Formula(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {}
    for name, value in state_dict.items():
        name = re.sub(r'(encoder\.layers\.\d+\.mlp)\.lin([12])\.', r'\1.fc\2.', name)
        for source, target in (('conv1', '0'), ('layer_norm1', '1'), ('conv2', '2'), ('layer_norm2', '3')):
            name = name.replace('encoder.neck.' + source + '.', 'encoder.neck.' + target + '.')
        name = name.replace('decoder.embed_tokens.weight', 'decoder.embed_tokens.emb.weight').replace('decoder.embed_positions.weight', 'decoder.embed_positions.emb.weight')
        mapped[name] = value
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, case=None):
    if case is None:
        raise ValueError('Formula recognition requires its pinned generation configuration')
    generation = case['reference']['generation_config']
    if generation.get('do_sample', False) or generation.get('num_beams', 1) != 1:
        raise ValueError('The published generation configuration uses greedy single-beam decoding')
    return {'generate': Workload(run=lambda: model.generate(inputs['pixel_values'], generation, case['generation_kwargs']['max_new_tokens']))}
