"""LXMERT's language, region and shared bidirectional cross-attention stacks."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from .mvp import EagerAttention
from ..runner import Workload


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.query = Linear(width, width)
        self.key = Linear(width, width)
        self.value = Linear(width, width)
        self.attention = EagerAttention(prescale_query=False, divide_scores=True)

    def forward(self, hidden, context, mask):
        shape = (*hidden.shape[:2], self.heads, hidden.shape[-1] // self.heads)
        query = self.query(hidden).view(shape)
        shape = (*context.shape[:2], self.heads, context.shape[-1] // self.heads)
        key, value = self.key(context).view(shape), self.value(context).view(shape)
        return self.attention(query, key, value, attn_mask=mask).reshape_as(hidden)


class AttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12, promote_fp32=False)

    def forward(self, hidden, residual):
        return self.LayerNorm(self.dense(hidden) + residual)


class AttentionLayer(nn.Module):
    def __init__(self, config, cross=False):
        super().__init__()
        self.cross = cross
        self.add_module('att' if cross else 'self', Attention(config))
        self.output = AttentionOutput(config)

    def forward(self, hidden, context=None, mask=None):
        attention = self.att if self.cross else self.self
        return self.output(attention(hidden, hidden if context is None else context, mask), hidden)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionLayer(config)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)

    def forward(self, hidden, mask):
        hidden = self.attention(hidden, mask=mask)
        return self.output(self.intermediate(hidden), hidden)


class CrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_attention = AttentionLayer(config, cross=True)
        self.lang_self_att, self.visn_self_att = AttentionLayer(config), AttentionLayer(config)
        self.lang_inter, self.visn_inter = EncoderIntermediate(config), EncoderIntermediate(config)
        self.lang_output, self.visn_output = EncoderOutput(config), EncoderOutput(config)

    def forward(self, language, vision, language_mask, vision_mask):
        # Both directions share parameters and consume the original pair.
        language, vision = (self.visual_attention(language, vision, vision_mask),
                            self.visual_attention(vision, language, language_mask))
        language = self.lang_self_att(language, mask=language_mask)
        vision = self.visn_self_att(vision, mask=vision_mask)
        return (self.lang_output(self.lang_inter(language), language),
                self.visn_output(self.visn_inter(vision), vision))


class LxmertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.embeddings = nn.Module()
        for name, size in (('word_embeddings', config.vocab_size),
                           ('position_embeddings', config.max_position_embeddings),
                           ('token_type_embeddings', config.type_vocab_size)):
            self.embeddings.add_module(name, Embedding(size, width, padding_idx=0))
        self.embeddings.LayerNorm = LayerNorm(width, eps=1e-12, promote_fp32=False)
        self.encoder = nn.Module()
        self.encoder.visn_fc = nn.Module()
        visual = self.encoder.visn_fc
        visual.visn_fc, visual.box_fc = Linear(config.visual_feat_dim, width), Linear(config.visual_pos_dim, width)
        visual.visn_layer_norm = LayerNorm(width, eps=1e-12, promote_fp32=False)
        visual.box_layer_norm = LayerNorm(width, eps=1e-12, promote_fp32=False)
        self.encoder.layer = nn.ModuleList([Layer(config) for _ in range(config.l_layers)])
        self.encoder.r_layers = nn.ModuleList([Layer(config) for _ in range(config.r_layers)])
        self.encoder.x_layers = nn.ModuleList([CrossLayer(config) for _ in range(config.x_layers)])
        self.pooler = nn.Module()
        self.pooler.dense, self.pooler.activation = Linear(width, width), Tanh()

    def forward(self, input_ids, visual_feats, visual_pos, attention_mask=None, token_type_ids=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        types = torch.zeros_like(input_ids) if token_type_ids is None else token_type_ids
        e = self.embeddings
        language = e.LayerNorm(e.word_embeddings(input_ids) + e.position_embeddings(positions) + e.token_type_embeddings(types))
        v = self.encoder.visn_fc
        vision = (v.visn_layer_norm(v.visn_fc(visual_feats)) + v.box_layer_norm(v.box_fc(visual_pos))) / 2
        mask = None
        if attention_mask is not None:
            mask = torch.zeros_like(attention_mask[:, None, None], dtype=language.dtype)
            mask.masked_fill_(~attention_mask[:, None, None].bool(), torch.finfo(language.dtype).min)
        for layer in self.encoder.layer:
            language = layer(language, mask)
        for layer in self.encoder.r_layers:
            vision = layer(vision, None)
        for layer in self.encoder.x_layers:
            language, vision = layer(language, vision, mask, None)
        return {'language_output': language, 'vision_output': vision,
                'pooled_output': self.pooler.activation(self.pooler.dense(language[:, 0]))}


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu' or config.layer_norm_eps != 1e-12:
        raise ValueError('LXMERT case preserves GELU and its native fixed normalization epsilon')
    return LxmertModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = {name: state_dict[name.replace('.emb.weight', '.weight')] for name in model.state_dict()}
    used = {name.replace('.emb.weight', '.weight') for name in mapped}
    if used != set(state_dict):
        raise KeyError(f'Unmapped LXMERT state: {sorted(set(state_dict) - used)}')
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
