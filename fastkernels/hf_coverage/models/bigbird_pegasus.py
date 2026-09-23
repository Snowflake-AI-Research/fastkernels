"""BigBird-Pegasus with the pinned evaluation sparse pattern and full decoder."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from .big_bird import EvalBlockSparseAttention
from .marian import load_state_dict_into as load_seq2seq
from .mbart import PreNormConditionalGeneration, PreNormStack
from .mvp import EagerAttention
from .plbart import make_workloads


class SparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.encoder_attention_heads
        self.dim = config.d_model // self.heads
        self.qkv = Linear(config.d_model, 3 * config.d_model, bias=config.use_bias)
        self.proj = Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.sparse = EvalBlockSparseAttention(config.block_size, config.num_random_blocks)

    def forward(self, hidden, attn_mask=None):
        batch, length = hidden.shape[:2]
        query, key, value = (tensor.reshape(batch, length, self.heads, self.dim).transpose(1, 2)
                             for tensor in self.qkv(hidden).chunk(3, dim=-1))
        context = self.sparse(query, key, value, attn_mask).transpose(1, 2).reshape(batch, length, -1)
        return self.proj(context)


class Encoder(PreNormStack):
    def __init__(self, config, shared):
        super().__init__(config, shared, decoder=False, learned_positions=False)
        self.embed_positions.emb.weight.requires_grad_(True)
        self.block, self.pad = config.block_size, config.pad_token_id
        for layer in self.layers:
            layer.attn = SparseAttention(config)
            layer.mlp.act = NewGELUActivation()

    def forward(self, ids, positions):
        hidden = self.embed_tokens(ids) * self.embed_scale + self.embed_positions(positions)
        batch, length = ids.shape
        padding = -length % self.block
        if padding:
            pad_ids = ids.new_full((batch, padding), self.pad)
            hidden = torch.cat((hidden, self.embed_tokens(pad_ids) * self.embed_scale), dim=1)
        mask = (torch.arange(length + padding, device=ids.device)[None] < length).expand(batch, -1)
        for layer in self.layers:
            hidden = layer(hidden, attn_mask=mask)
        return self.layer_norm(hidden)[:, :length]


def build_from_config(config, device, dtype):
    if (config.attention_type != "block_sparse" or config.activation_function != "gelu_new"
            or config.use_bias or not config.use_cache or not config.tie_word_embeddings):
        raise ValueError("Selected BigBird-Pegasus requires block sparse, GELU-new, bias-free attention, tied output and caching")
    model = PreNormConditionalGeneration(config, learned_positions=False)
    model.encoder = Encoder(config, model.shared)
    model.decoder.embed_positions.emb.weight.requires_grad_(True)
    for layer in model.decoder.layers:
        layer.intermediate.intermediate_act_fn = NewGELUActivation()
        layer.attention.self.qkv.bias = None
        layer.attention.output.dense.bias = None
        layer.attention.self.attn = EagerAttention(prescale_query=False)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            getattr(layer.cross_attention, name).bias = None
        layer.cross_attention.attention = EagerAttention(prescale_query=False)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    translated = {}
    for name, value in state_dict.items():
        name = name.replace(".layernorm_embedding.", ".layer_norm.")
        for original, target in (("query", "q"), ("key", "k"), ("value", "v")):
            name = name.replace(f".self_attn.self.{original}.", f".self_attn.{target}_proj.")
        name = name.replace(".self_attn.output.", ".self_attn.out_proj.")
        translated[name] = value
    load_seq2seq(model, translated, config)
