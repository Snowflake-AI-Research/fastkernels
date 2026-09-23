"""MarkupLM's text and XPath embeddings followed by the existing BERT encoder."""

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from ..runner import Workload
from .layoutlm import LayoutAttention


class MarkupLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding = config.pad_token_id
        self.words = Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding)
        self.positions = Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding)
        self.types = Embedding(config.type_vocab_size, config.hidden_size)
        self.tags = nn.ModuleList([Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size)
                                   for _ in range(config.max_depth)])
        self.subscripts = nn.ModuleList([Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size)
                                         for _ in range(config.max_depth)])
        self.path_inner = Linear(config.max_depth * config.xpath_unit_hidden_size, 4 * config.hidden_size)
        self.path_out = Linear(4 * config.hidden_size, config.hidden_size)
        self.relu = ReLU()
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.encoder = BertEncoder(config)
        # Native MarkupLM stores BF16 scores and evaluates softmax in FP32.
        # CLIP's existing attention, reused by LayoutAttention, preserves both.
        for layer in self.encoder.layer:
            layer.attention = LayoutAttention(config)
        self.pooler = Linear(config.hidden_size, config.hidden_size)
        self.tanh = Tanh()

    def forward(self, input_ids, xpath_tags_seq, xpath_subs_seq):
        mask = input_ids.ne(self.padding).to(torch.int64)
        positions = mask.cumsum(-1) * mask + self.padding
        tags = torch.cat([table(xpath_tags_seq[:, :, depth]) for depth, table in enumerate(self.tags)], dim=-1)
        subs = torch.cat([table(xpath_subs_seq[:, :, depth]) for depth, table in enumerate(self.subscripts)], dim=-1)
        path = self.path_out(self.relu(self.path_inner(tags + subs)))
        hidden = self.words(input_ids) + self.positions(positions) + self.types(torch.zeros_like(input_ids)) + path
        hidden = self.encoder(self.norm(hidden))
        return {'last_hidden_state': hidden, 'pooler_output': self.tanh(self.pooler(hidden[:, 0]))}


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu':
        raise ValueError('Selected MarkupLM computation uses the ordinary GELU encoder')
    return MarkupLMModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, weights = dict(state_dict), {}
    # Pinned XPathEmbeddings registers this unused older projection.
    remaining.pop('embeddings.xpath_embeddings.xpath_unitseq2_embeddings.weight')
    remaining.pop('embeddings.xpath_embeddings.xpath_unitseq2_embeddings.bias')
    names = {'words': 'embeddings.word_embeddings', 'positions': 'embeddings.position_embeddings',
             'types': 'embeddings.token_type_embeddings', 'path_inner': 'embeddings.xpath_embeddings.xpath_unitseq2_inner',
             'path_out': 'embeddings.xpath_embeddings.inner2emb', 'norm': 'embeddings.LayerNorm', 'pooler': 'pooler.dense'}
    for name in model.state_dict():
        if name.startswith('tags.') or name.startswith('subscripts.'):
            kind, index, _, field = name.split('.')
            prefix = 'xpath_tag_sub_embeddings' if kind == 'tags' else 'xpath_subs_sub_embeddings'
            source = f'embeddings.xpath_embeddings.{prefix}.{index}.{field}'
        elif name.startswith('encoder.'):
            source = name
            for projection, native in (('q', 'query'), ('k', 'key'), ('v', 'value')):
                source = source.replace(f'.attention.core.{projection}_proj.', f'.attention.self.{native}.')
            source = source.replace('.attention.core.out_proj.', '.attention.output.dense.')
            source = source.replace('.attention.LayerNorm.', '.attention.output.LayerNorm.')
        else:
            source = names[name.split('.')[0]] + '.' + name.split('.')[-1]
        weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped MarkupLM parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def make_workloads(model, inputs, config):
    return {'forward': Workload(run=lambda: model(**inputs))}
