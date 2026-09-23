"""TAPAS table embeddings, per-cell positions and the existing BERT encoder."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from ..runner import Workload
from .mvp import EagerAttention


class TapasModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_positions = config.max_position_embeddings
        self.columns, self.rows = config.type_vocab_sizes[1:3]
        self.words = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.positions = Embedding(config.max_position_embeddings, config.hidden_size)
        self.types = nn.ModuleList([Embedding(size, config.hidden_size) for size in config.type_vocab_sizes])
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.encoder = BertEncoder(config)
        # Native TAPAS rounds QK scores before division and BF16 softmax.
        for layer in self.encoder.layer:
            layer.attention.self.attn = EagerAttention(prescale_query=False, divide_scores=True)
        self.pooler = Linear(config.hidden_size, config.hidden_size)
        self.tanh = Tanh()

    def forward(self, input_ids, token_type_ids):
        batch, length = input_ids.shape
        absolute = torch.arange(length, device=input_ids.device)[None].expand(batch, -1)
        cell = token_type_ids[:, :, 1] * self.rows + token_type_ids[:, :, 2]
        # This reduction concerns only integer table/position metadata, not activations.
        first = torch.full((batch, self.columns * self.rows), length, dtype=torch.long, device=input_ids.device)
        first.scatter_reduce_(1, cell, absolute, reduce='amin', include_self=True)
        positions = (absolute - first.gather(1, cell)).clamp(max=self.max_positions - 1)
        hidden = self.words(input_ids) + self.positions(positions)
        for channel, embedding in enumerate(self.types):
            hidden = hidden + embedding(token_type_ids[:, :, channel])
        hidden = self.encoder(self.norm(hidden))
        return {'last_hidden_state': hidden, 'pooler_output': self.tanh(self.pooler(hidden[:, 0]))}


def build_from_config(config, device, dtype):
    if not config.reset_position_index_per_cell or config.hidden_act != 'gelu' or config.is_decoder:
        raise ValueError('The selected TAPAS default uses per-cell positions and a GELU encoder')
    return TapasModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    names = {'words': 'embeddings.word_embeddings', 'positions': 'embeddings.position_embeddings',
             'norm': 'embeddings.LayerNorm', 'pooler': 'pooler.dense'}
    for name in model.state_dict():
        if name.startswith('types.'):
            source = f'embeddings.token_type_embeddings_{name.split(".")[1]}.weight'
        elif name.startswith('encoder.'):
            source = name
            if '.qkv.' in name:
                weights[name] = torch.cat([remaining.pop(name.replace('.qkv.', f'.{part}.')) for part in ('query', 'key', 'value')])
                continue
        else:
            source = names[name.split('.')[0]] + '.' + name.split('.')[-1]
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped TAPAS parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
