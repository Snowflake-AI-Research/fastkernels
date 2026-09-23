"""BioGPT's scaled embeddings and pre-normalized causal decoder."""

import math

import torch
from torch import nn
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from .llama import make_workloads
from .qwen2_precision import DenseCachedAttention


class DecoderLayer(nn.Module):
    def __init__(self, config, attention_class=LlamaAttention):
        super().__init__()
        if attention_class is LlamaAttention:
            self.self_attn = attention_class(config.hidden_size, config.num_attention_heads,
                                             config.num_attention_heads, config.hidden_size // config.num_attention_heads,
                                             bias=True, o_proj_bias=True, nope=True)
        else:
            self.self_attn = attention_class(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)
        self.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size)

    def forward(self, hidden, positions):
        hidden = hidden + self.self_attn(positions, self.self_attn_layer_norm(hidden))
        return hidden + self.mlp(self.final_layer_norm(hidden))


class Decoder(nn.Module):
    def __init__(self, config, attention_class=LlamaAttention):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.embed_positions = Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        self.layers = nn.ModuleList([DecoderLayer(config, attention_class) for _ in range(config.num_hidden_layers)])
        self.layer_norm = LayerNorm(config.hidden_size, eps=1e-5, promote_fp32=False)

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids) * self.embed_scale + self.embed_positions(positions + 2)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.layer_norm(hidden)


class BioGptForCausalLM(nn.Module):
    def __init__(self, config, attention_class=LlamaAttention):
        super().__init__()
        self.model = Decoder(config, attention_class)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight


def build_from_config(config, device, dtype):
    if config.hidden_act != 'gelu' or not config.use_cache:
        raise ValueError('BioGPT case retains GELU and default cached decoding')
    model = BioGptForCausalLM(config)
    for layer in model.model.layers:
        layer.self_attn.attn = DenseCachedAttention(
            config.num_attention_heads, config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )
    return model.to(device=device, dtype=dtype).eval()


def load_decoder_weights(model, state_dict, config, *, prefix, head, sinusoidal=False):
    if config.tie_word_embeddings and not torch.equal(state_dict[head + '.weight'], state_dict[prefix + '.embed_tokens.weight']):
        raise ValueError('Tied decoder input and output weights disagree')
    remaining = dict(state_dict)
    weights = {}
    for name in model.state_dict():
        source = name.replace('model.', prefix + '.', 1)
        source = source.replace('.embedding_op.emb.weight', '.weight').replace('.emb.weight', '.weight')
        source = source.replace('.mlp.fc1.', '.fc1.').replace('.mlp.fc2.', '.fc2.')
        source = source.replace('.self_attn.o_proj.', '.self_attn.out_proj.')
        if name.startswith('lm_head.'):
            source = head + '.weight'
        if sinusoidal and name == 'model.embed_positions.emb.weight':
            weights[name] = model.model.embed_positions.emb.weight
            continue
        if '.qkv_proj.' in source:
            weights[name] = torch.cat([remaining.pop(source.replace('qkv_proj', part + '_proj'))
                                       for part in ('q', 'k', 'v')])
        else:
            weights[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f'Unmapped decoder parameters: {sorted(remaining)}')
    model.load_state_dict(weights, strict=True)


def load_state_dict_into(model, state_dict, config):
    load_decoder_weights(model, state_dict, config, prefix='biogpt', head='output_projection')
