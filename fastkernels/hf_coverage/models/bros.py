"""BROS document positions and attention composed from existing operations."""

import math
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from ..runner import Workload


class BoxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.num_attention_heads, config.dim_bbox_projection
        for name in ('query', 'key', 'value'):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def forward(self, hidden, box_positions):
        batch, length, _ = hidden.shape
        q, k, v = [projection(hidden).view(batch, length, self.heads, self.width).transpose(1, 2)
                   for projection in (self.query, self.key, self.value)]
        q_flat, k_flat, v_flat = [x.reshape(batch * self.heads, length, self.width) for x in (q, k, v)]
        scores = self.bmm(q_flat, k_flat.transpose(1, 2)).view(batch, self.heads, length, length)
        position_scores = self.bmm(q.transpose(1, 2).reshape(batch * length, self.heads, self.width),
                                  box_positions.reshape(batch * length, length, self.width).transpose(1, 2))
        scores = (scores + position_scores.view(batch, length, self.heads, length).transpose(1, 2)) / math.sqrt(self.width)
        context = self.bmm(self.softmax(scores).reshape(batch * self.heads, length, length), v_flat)
        return context.view(batch, self.heads, length, self.width).transpose(1, 2).reshape_as(hidden)


class BrosLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn, self.attn_output = BoxAttention(config), EncoderSelfOutput(config)
        self.intermediate, self.output = EncoderIntermediate(config), EncoderOutput(config)

    def forward(self, hidden, box_positions):
        hidden = self.attn_output(self.attn(hidden, box_positions), hidden)
        return self.output(self.intermediate(hidden), hidden)


class BrosModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bbox_scale = config.bbox_scale
        self.words = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.positions = Embedding(config.max_position_embeddings, config.hidden_size)
        self.types = Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings)[None])
        self.register_buffer('x_frequency', torch.empty(config.dim_bbox_sinusoid_emb_1d // 2))
        self.register_buffer('y_frequency', torch.empty(config.dim_bbox_sinusoid_emb_1d // 2))
        self.box_projection = Linear(config.dim_bbox_sinusoid_emb_2d, config.dim_bbox_projection, bias=False)
        self.layers = nn.ModuleList([BrosLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler, self.tanh = Linear(config.hidden_size, config.hidden_size), Tanh()

    def forward(self, input_ids, bbox):
        # Boxes are supplied position metadata, independent of hidden activations.
        corners = bbox[..., [0, 1, 2, 1, 2, 3, 0, 3]] * self.bbox_scale
        differences = corners[:, None] - corners[:, :, None]
        features = []
        for coordinate in range(8):
            frequency = self.x_frequency if coordinate % 2 == 0 else self.y_frequency
            phase = differences[..., coordinate, None] * frequency
            features.append(torch.cat((phase.sin(), phase.cos()), dim=-1))
        box_positions = self.box_projection(torch.cat(features, dim=-1))
        hidden = self.words(input_ids) + self.types(torch.zeros_like(input_ids))
        hidden = self.norm(hidden + self.positions(self.position_ids[:, :input_ids.shape[1]]))
        for layer in self.layers:
            hidden = layer(hidden, box_positions)
        return {'last_hidden_state': hidden, 'pooler_output': self.tanh(self.pooler(hidden[:, 0]))}


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu' or config.is_decoder:
        raise ValueError('Selected BROS configuration requires its ordinary GELU encoder')
    return BrosModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    prefixes = {'words': 'embeddings.word_embeddings', 'positions': 'embeddings.position_embeddings',
                'types': 'embeddings.token_type_embeddings', 'norm': 'embeddings.LayerNorm',
                'box_projection': 'bbox_embeddings.bbox_projection', 'pooler': 'pooler.dense'}
    buffers = {'position_ids': 'embeddings.position_ids',
               'x_frequency': 'bbox_embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq',
               'y_frequency': 'bbox_embeddings.bbox_sinusoid_emb.y_pos_emb.inv_freq'}
    for name in model.state_dict():
        if name in buffers:
            source = buffers[name]
        elif name.startswith('layers.'):
            source = name.replace('layers.', 'encoder.layer.', 1).replace('.attn.', '.attention.self.')
            source = source.replace('.attn_output.', '.attention.output.')
        else:
            prefix, field = name.split('.')[0], name.split('.')[-1]
            source = prefixes[prefix] + '.' + field
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped BROS parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
