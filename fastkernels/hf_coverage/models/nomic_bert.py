"""NomicBERT masked LM composed from bidirectional rotary attention and SwiGLU."""

from types import SimpleNamespace

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.silu import SiLU
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L2.llada_attention import LLaDAAttention
from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP
from .bert import MaskedLMHead, make_workloads


class NomicLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.self_attn = LLaDAAttention(config.hidden_size, config.num_attention_heads,
                                      config.num_attention_heads, config.head_dim, rotary_emb=rotary)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)
        self.mlp = LlamaMLP(config)
        self.post_mlp_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, hidden):
        attended, _ = self.self_attn(hidden)
        hidden = self.post_attention_layernorm(hidden + attended)
        return self.post_mlp_layernorm(hidden + self.mlp(hidden))


class NomicBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = SimpleNamespace(**(dict(config) | {'position_embedding_type': 'rotary'}))
        self.embeddings = BertEmbeddings(embedding_config)
        del self.embeddings.position_embeddings
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings,
                                      config.rope_parameters['rope_theta'])
        self.layers = nn.ModuleList([NomicLayer(config, self.rotary) for _ in range(config.num_hidden_layers)])
        self.lm_head = MaskedLMHead(config)
        self.lm_head.activation = SiLU()
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        positions = self.embeddings.position_ids[:, :input_ids.shape[1]]
        hidden = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(hidden)


def build_from_config(config, device, dtype):
    if config.hidden_act != 'silu' or config.rope_parameters['rope_type'] != 'default':
        raise ValueError('NomicBERT case preserves default SwiGLU and static rotary embeddings')
    return NomicBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    used = set()
    for name in model.state_dict():
        if name.startswith('embeddings.'):
            source = 'nomic_bert.' + name.replace('.emb.weight', '.weight')
        elif name.startswith('layers.'):
            source = 'nomic_bert.' + name.replace('.attn_out.', '.o_proj.')
            if '.gate_up_proj.' in source:
                sources = [source.replace('gate_up_proj', part) for part in ('gate_proj', 'up_proj')]
                weights[name] = torch.cat([state_dict[key] for key in sources])
                used.update(sources)
                continue
        else:
            local = name.removeprefix('lm_head.')
            if local.startswith('decoder.'):
                source = 'cls.predictions.' + local
            else:
                source = 'cls.predictions.transform.' + local.replace('LayerNorm.', 'layer_norm.')
        weights[name] = state_dict[source]
        used.add(source)
    if not torch.equal(state_dict['cls.predictions.bias'], state_dict['cls.predictions.decoder.bias']):
        raise ValueError('Masked-LM bias aliases disagree')
    used.add('cls.predictions.bias')
    if getattr(config, "tie_word_embeddings", True) and not torch.equal(state_dict['cls.predictions.decoder.weight'], state_dict['nomic_bert.embeddings.word_embeddings.weight']):
        raise ValueError('Tied masked-LM weights disagree')
    if set(state_dict) != used:
        raise KeyError(f'Unmapped NomicBERT weights: {sorted(set(state_dict) - used)}')
    model.load_state_dict(weights, strict=True)
