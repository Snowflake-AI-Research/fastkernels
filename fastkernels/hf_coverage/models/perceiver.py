"""The documented bare PerceiverModel constructor, with latent outputs."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax


class Attention(nn.Module):
    def __init__(self, config, cross):
        super().__init__()
        self.heads = config.num_cross_attention_heads if cross else config.num_self_attention_heads
        qdim, kdim = config.d_latents, config.d_model if cross else config.d_latents
        qk = config.qk_channels
        if qk is None:
            qk = kdim if cross and config.cross_attention_shape_for_attention == "kv" else qdim
        value = config.v_channels or qk
        self.query_dim, self.value_dim = qk // self.heads, value // self.heads
        self.layernorm1 = LayerNorm(qdim, promote_fp32=False)
        self.layernorm2 = LayerNorm(kdim, promote_fp32=False) if cross else nn.Identity()
        self.query, self.key, self.value = Linear(qdim, qk), Linear(kdim, qk), Linear(kdim, value)
        self.bmm, self.softmax = BMM(), Softmax(dim=-1)

    def forward(self, hidden, inputs=None):
        hidden = self.layernorm1(hidden)
        source = self.layernorm2(inputs) if inputs is not None else hidden
        batch = hidden.shape[0]
        query = self.query(hidden).reshape(batch, -1, self.heads, self.query_dim).transpose(1, 2)
        key = self.key(source).reshape(batch, -1, self.heads, self.query_dim).transpose(1, 2)
        value = self.value(source).reshape(batch, -1, self.heads, self.value_dim).transpose(1, 2)
        scores = self.bmm(query, key.transpose(-1, -2)) / self.query_dim ** 0.5
        context = self.bmm(self.softmax(scores), value)
        return context.transpose(1, 2).reshape(batch, hidden.shape[1], -1)


class Layer(nn.Module):
    def __init__(self, config, cross):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = Attention(config, cross)
        self.attention.output = nn.Module()
        width = config.d_latents
        attention = self.attention.self
        self.attention.output.dense = Linear(attention.heads * attention.value_dim, width)
        self.layernorm = LayerNorm(width, promote_fp32=False)
        factor = config.cross_attention_widening_factor if cross else config.self_attention_widening_factor
        self.mlp = nn.Module()
        self.mlp.dense1, self.mlp.dense2 = Linear(width, factor * width), Linear(factor * width, width)
        self.activation = GELU()
        self.residual = config.use_query_residual if cross else True

    def forward(self, hidden, inputs=None):
        attention = self.attention.output.dense(self.attention.self(hidden, inputs))
        if self.residual:
            attention = attention + hidden
        return attention + self.mlp.dense2(self.activation(self.mlp.dense1(self.layernorm(attention))))


class PerceiverModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.latents = nn.Parameter(torch.empty(config.num_latents, config.d_latents))
        self.encoder = nn.Module()
        self.encoder.cross_attention = Layer(config, True)
        self.encoder.self_attends = nn.ModuleList([Layer(config, False) for _ in range(config.num_self_attends_per_block)])
        self.blocks = config.num_blocks

    def forward(self, inputs):
        hidden = self.embeddings.latents.expand(inputs.shape[0], -1, -1)
        hidden = self.encoder.cross_attention(hidden, inputs)
        for _ in range(self.blocks):
            for layer in self.encoder.self_attends:
                hidden = layer(hidden)
        return {"last_hidden_state": hidden}


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.chunk_size_feed_forward:
        raise ValueError("The selected Perceiver constructor uses GELU without feed-forward chunking")
    return PerceiverModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    model.load_state_dict(state_dict, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
