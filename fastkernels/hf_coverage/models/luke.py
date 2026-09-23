"""LUKE's entity-aware attention composed from existing operations."""

import math
import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.embedding_bag import EmbeddingBag
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.bmm import BatchMatMul
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from ..runner import Workload


class EntityAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        for name in ('query', 'key', 'value', 'w2e_query', 'e2w_query', 'e2e_query'):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
        self.bmm, self.softmax = BatchMatMul(), Softmax()

    def split(self, tensor):
        return tensor.view(tensor.shape[0], tensor.shape[1], self.heads, self.width).transpose(1, 2)

    def multiply(self, left, right):
        batch, heads = left.shape[:2]
        result = self.bmm(left.reshape(batch * heads, *left.shape[2:]),
                          right.reshape(batch * heads, *right.shape[2:]))
        return result.view(batch, heads, *result.shape[1:])

    def forward(self, hidden, word_count):
        word, entity = hidden[:, :word_count], hidden[:, word_count:]
        keys, values = self.split(self.key(hidden)), self.split(self.value(hidden))
        word_keys, entity_keys = keys[:, :, :word_count], keys[:, :, word_count:]
        word_scores = torch.cat((
            self.multiply(self.split(self.query(word)), word_keys.transpose(-1, -2)),
            self.multiply(self.split(self.w2e_query(word)), entity_keys.transpose(-1, -2))), dim=-1)
        entity_scores = torch.cat((
            self.multiply(self.split(self.e2w_query(entity)), word_keys.transpose(-1, -2)),
            self.multiply(self.split(self.e2e_query(entity)), entity_keys.transpose(-1, -2))), dim=-1)
        scores = torch.cat((word_scores, entity_scores), dim=-2) / math.sqrt(self.width)
        context = self.multiply(self.softmax(scores), values)
        return context.transpose(1, 2).reshape_as(hidden)


class LukeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = EntityAttention(config)
        self.attn_output = EncoderSelfOutput(config)
        self.intermediate, self.output = EncoderIntermediate(config), EncoderOutput(config)

    def forward(self, hidden, word_count):
        hidden = self.attn_output(self.attn(hidden, word_count), hidden)
        return self.output(self.intermediate(hidden), hidden)


class LukeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding = config.pad_token_id
        self.words = Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding)
        self.word_positions = Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding)
        self.word_types = Embedding(config.type_vocab_size, config.hidden_size)
        self.word_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.entities = Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        self.entity_projection = (Linear(config.entity_emb_size, config.hidden_size, bias=False)
                                  if config.entity_emb_size != config.hidden_size else nn.Identity())
        self.entity_positions = EmbeddingBag(config.max_position_embeddings, config.hidden_size,
                                            mode='sum', include_last_offset=True)
        self.entity_types = Embedding(config.type_vocab_size, config.hidden_size)
        self.entity_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.layers = nn.ModuleList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler, self.tanh = Linear(config.hidden_size, config.hidden_size), Tanh()

    def forward(self, input_ids, entity_ids, entity_position_ids):
        nonpadding = input_ids.ne(self.padding).long()
        positions = nonpadding.cumsum(-1) * nonpadding + self.padding
        word = self.word_norm(self.words(input_ids) + self.word_positions(positions)
                              + self.word_types(torch.zeros_like(input_ids)))
        # Group the valid mention positions; only these integer indices are reduced here.
        valid = entity_position_ids.ne(-1)
        counts = valid.sum(-1).flatten()
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        # HF rounds the sum to the model dtype before dividing. EmbeddingBag's
        # mean mode skips this boundary. Counts depend only on supplied masks.
        summed = self.entity_positions(entity_position_ids[valid], offsets)
        averaged = (summed / counts.clamp(min=1).unsqueeze(-1)).view(*entity_ids.shape, -1)
        entity = self.entity_norm(self.entity_projection(self.entities(entity_ids)) + averaged
                                  + self.entity_types(torch.zeros_like(entity_ids)))
        hidden = torch.cat((word, entity), dim=1)
        for layer in self.layers:
            hidden = layer(hidden, input_ids.shape[1])
        word, entity = hidden[:, :input_ids.shape[1]], hidden[:, input_ids.shape[1]:]
        return {'last_hidden_state': word, 'entity_last_hidden_state': entity,
                'pooler_output': self.tanh(self.pooler(word[:, 0]))}


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu' or not config.use_entity_aware_attention:
        raise ValueError('Selected LUKE configuration requires GELU and entity-aware attention')
    return LukeModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    prefixes = {'words': 'embeddings.word_embeddings', 'word_positions': 'embeddings.position_embeddings',
                'word_types': 'embeddings.token_type_embeddings', 'word_norm': 'embeddings.LayerNorm',
                'entities': 'entity_embeddings.entity_embeddings',
                'entity_projection': 'entity_embeddings.entity_embedding_dense',
                'entity_positions': 'entity_embeddings.position_embeddings',
                'entity_types': 'entity_embeddings.token_type_embeddings',
                'entity_norm': 'entity_embeddings.LayerNorm', 'pooler': 'pooler.dense'}
    for name in model.state_dict():
        if name.startswith('layers.'):
            source = name.replace('layers.', 'encoder.layer.', 1).replace('.attn.', '.attention.self.')
            source = source.replace('.attn_output.', '.attention.output.')
        else:
            prefix, field = name.split('.')[0], name.split('.')[-1]
            source = prefixes[prefix] + '.' + field
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped LUKE parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
