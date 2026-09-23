"""MegatronBERT masked LM using existing pre-normalized ViT blocks."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.vit_encoder_block import VitEncoderBlock
from .bert import MaskedLMHead, make_workloads
from .rembert import RoundedEncoderAttention
from .roberta_prelayernorm import PreNormAttention


class MegatronBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.embeddings.LayerNorm = nn.Identity()
        self.layers = nn.ModuleList([
            VitEncoderBlock(config.hidden_size, config.num_attention_heads,
                            mlp_ratio=config.intermediate_size / config.hidden_size,
                            norm_eps=config.layer_norm_eps)
            for _ in range(config.num_hidden_layers)
        ])
        for layer in self.layers:
            layer.attn = PreNormAttention(config)
            # HF stores scores and probabilities in the execution dtype.
            layer.attn.attn = RoundedEncoderAttention()
        self.final_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        positions = self.embeddings.position_ids[:, :input_ids.shape[1]]
        hidden = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.final_norm(hidden))


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu' or config.is_decoder or config.add_cross_attention:
        raise ValueError('MegatronBERT case preserves its default bidirectional GELU computation')
    return MegatronBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    names = {'norm1': 'attention.ln', 'norm2': 'ln', 'attn.proj': 'attention.output.dense',
             'mlp.fc1': 'intermediate.dense', 'mlp.fc2': 'output.dense'}
    weights = {}
    used = set()
    for name in model.state_dict():
        if name.startswith('lm_head.'):
            continue
        if name.startswith('layers.'):
            _, index, rest = name.split('.', 2)
            module, field = rest.rsplit('.', 1)
            prefix = f'bert.encoder.layer.{index}.'
            if module == 'attn.qkv':
                sources = [prefix + f'attention.self.{part}.{field}' for part in ('query', 'key', 'value')]
                weights[name] = torch.cat([state_dict[source] for source in sources])
                used.update(sources)
                continue
            source = prefix + names[module] + '.' + field
        elif name.startswith('final_norm.'):
            source = 'bert.encoder.ln.' + name.split('.')[-1]
        else:
            source = 'bert.' + name.replace('.emb.weight', '.weight')
        weights[name] = state_dict[source]
        used.add(source)
    for name in model.lm_head.state_dict():
        prefix = 'cls.predictions.' if name.startswith('decoder.') else 'cls.predictions.transform.'
        weights['lm_head.' + name] = state_dict[prefix + name]
        used.add(prefix + name)
    if not torch.equal(state_dict['cls.predictions.bias'], state_dict['cls.predictions.decoder.bias']):
        raise ValueError('Masked-LM bias aliases disagree')
    used.add('cls.predictions.bias')
    if config.tie_word_embeddings and not torch.equal(state_dict['cls.predictions.decoder.weight'], state_dict['bert.embeddings.word_embeddings.weight']):
        raise ValueError('Tied masked-LM weights disagree')
    if set(state_dict) != used:
        raise KeyError(f'Unmapped MegatronBERT weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(weights, strict=True)
