"""CANINE's character, molecule and reconstructed-character encoder stages."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.conv1d import Conv1d
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L1.tensor_ops import Pad
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer
from ..patches.hashed_embedding import HashedEmbedding
from ..runner import Workload


class CanineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.buckets, self.stride = config.num_hash_buckets, config.local_transformer_stride
        self.rate, self.kernel = config.downsampling_rate, config.upsampling_kernel_size
        primes = (31, 43, 59, 61, 73, 97, 103, 113)
        self.hash_embeddings = nn.ModuleList([
            HashedEmbedding(self.buckets, config.hidden_size // config.num_hash_functions, prime)
            for prime in primes
        ])
        self.positions = Embedding(self.buckets, config.hidden_size)
        self.types = Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.initial = BertLayer(config)
        self.downsample = Conv1d(config.hidden_size, config.hidden_size, self.rate, stride=self.rate)
        self.down_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = Tanh()
        self.upsample = Conv1d(2 * config.hidden_size, config.hidden_size, self.kernel)
        self.up_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.gelu, self.pad = GELU(), Pad()
        self.final = BertLayer(config)

    def forward(self, input_ids):
        length = input_ids.shape[1]
        hidden = torch.cat([embedding(input_ids) for embedding in self.hash_embeddings], -1)
        hidden = hidden + self.types(torch.zeros_like(input_ids))
        hidden = self.embed_norm(hidden + self.positions(torch.arange(length, device=input_ids.device)))
        chars = torch.cat([self.initial(hidden[:, start:start + self.stride])
                           for start in range(0, length, self.stride)], dim=1)
        down = self.gelu(self.downsample(chars.transpose(1, 2)).transpose(1, 2))
        molecules = self.down_norm(torch.cat((chars[:, :1], down[:, :-1]), dim=1))
        for layer in self.layers:
            molecules = layer(molecules)
        pooled = self.pooler_activation(self.pooler(molecules[:, 0]))
        repeated = torch.cat((molecules[:, 1:].repeat_interleave(self.rate, dim=1),
                              molecules[:, -1:].repeat_interleave(length % self.rate + self.rate, dim=1)), dim=1)
        merged = torch.cat((chars, repeated), dim=-1).transpose(1, 2)
        left = (self.kernel - 1) // 2
        projected = self.upsample(self.pad(merged, (left, self.kernel - 1 - left))).transpose(1, 2)
        hidden = self.final(self.up_norm(self.gelu(projected)))
        return {'last_hidden_state': hidden, 'pooler_output': pooled}


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu' or config.num_hash_functions != 8:
        raise ValueError('Selected CANINE default requires GELU and all eight character hash tables')
    return CanineModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    direct = {'positions.emb.weight': 'char_embeddings.char_position_embeddings.weight',
              'types.emb.weight': 'char_embeddings.token_type_embeddings.weight'}
    prefixes = {'embed_norm': 'char_embeddings.LayerNorm', 'downsample': 'chars_to_molecules.conv',
                'down_norm': 'chars_to_molecules.LayerNorm', 'upsample': 'projection.conv',
                'up_norm': 'projection.LayerNorm', 'pooler': 'pooler.dense'}
    for name in model.state_dict():
        if name.startswith('hash_embeddings.'):
            source = f'char_embeddings.HashBucketCodepointEmbedder_{name.split(".")[1]}.weight'
        elif name in direct:
            source = direct[name]
        elif name.split('.')[0] in prefixes:
            part, field = name.split('.')[0], name.split('.')[-1]
            source = prefixes[part] + '.' + field
        else:
            if name.startswith('initial.'):
                source = name.replace('initial.', 'initial_char_encoder.layer.0.', 1)
            elif name.startswith('final.'):
                source = name.replace('final.', 'final_char_encoder.layer.0.', 1)
            else:
                source = name.replace('layers.', 'encoder.layer.', 1)
            if '.qkv.' in source:
                weights[name] = torch.cat([remaining.pop(source.replace('.qkv.', f'.{part}.')) for part in ('query', 'key', 'value')])
                continue
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped CANINE parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(inputs['input_ids']))}
