"""The documented ESM-1b masked-LM path, with learned absolute positions."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from .bert import make_workloads


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.width = config.hidden_size // self.heads
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = Linear(config.hidden_size, config.hidden_size)
        self.attention = DenseAttention(backend='sdpa')
        self.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size)

    def forward(self, hidden):
        batch, length = hidden.shape[:2]
        q, k, v = [part.reshape(batch, length, self.heads, self.width)
                   for part in self.qkv(self.norm1(hidden)).chunk(3, -1)]
        context = self.attention(q * self.width**-0.5, k, v, softmax_scale=1.0)
        hidden = hidden + self.proj(context.reshape(batch, length, -1))
        return hidden + self.mlp(self.norm2(hidden))


class EsmForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.head_dense = Linear(config.hidden_size, config.hidden_size)
        self.head_gelu = GELU()
        self.head_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.decoder = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.empty(config.vocab_size))
        if config.tie_word_embeddings:
            self.decoder.weight = self.word_embeddings.emb.weight

    def forward(self, input_ids):
        nonpadding = input_ids.ne(self.padding_idx).to(torch.int64)
        positions = nonpadding.cumsum(-1) * nonpadding + self.padding_idx
        hidden = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.head_norm(self.head_gelu(self.head_dense(self.final_norm(hidden))))
        return self.decoder(hidden) + self.bias


def build_from_config(config, device, dtype):
    if (config.position_embedding_type != 'absolute' or config.token_dropout
            or config.emb_layer_norm_before or config.is_decoder or config.add_cross_attention):
        raise ValueError('The documented facebook/esm-1b config selects absolute positions without token dropout')
    return EsmForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    # The separate contact-prediction method is not called by the MLM task.
    for name in ('esm.contact_head.regression.weight', 'esm.contact_head.regression.bias'):
        remaining.pop(name)
    weights = {}
    layer_names = {'norm1': 'attention.LayerNorm', 'norm2': 'LayerNorm',
                   'proj': 'attention.output.dense', 'mlp.fc1': 'intermediate.dense',
                   'mlp.fc2': 'output.dense'}
    names = {'word_embeddings.emb.weight': 'esm.embeddings.word_embeddings.weight',
             'position_embeddings.emb.weight': 'esm.embeddings.position_embeddings.weight',
             'bias': 'lm_head.bias', 'decoder.weight': 'lm_head.decoder.weight'}
    for name in model.state_dict():
        if name.startswith('layers.'):
            _, index, rest = name.split('.', 2)
            part, field = rest.rsplit('.', 1)
            prefix = f'esm.encoder.layer.{index}.'
            if part == 'qkv':
                weights[name] = torch.cat([remaining.pop(prefix + f'attention.self.{p}.{field}') for p in ('query', 'key', 'value')])
                continue
            source = prefix + layer_names[part] + '.' + field
        elif name in names:
            source = names[name]
        else:
            part, field = name.split('.')
            source = {'final_norm': 'esm.encoder.emb_layer_norm_after', 'head_dense': 'lm_head.dense',
                      'head_norm': 'lm_head.layer_norm'}[part] + '.' + field
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped ESM weights: {sorted(remaining)}')
    if config.tie_word_embeddings and not torch.equal(weights['decoder.weight'], weights['word_embeddings.emb.weight']):
        raise ValueError('Tied ESM weights disagree')
    model.load_state_dict(weights, strict=True)
