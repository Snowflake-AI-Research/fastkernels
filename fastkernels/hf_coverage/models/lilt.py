"""LiLT's coupled text/layout streams and default text pooler."""

import copy
import math

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfOutput
from fastkernels.tasks.baseline.L2.encoder_embeddings import (
    XLMRobertaEmbeddings, create_roberta_position_ids_from_input_ids,
)
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate, EncoderOutput

from .layoutlm import BoxEmbeddings, Pooler, make_workloads


class LayoutEmbeddings(BoxEmbeddings):
    def __init__(self, config):
        super().__init__(config, config.hidden_size // 6)
        width = config.hidden_size // config.channel_shrink_ratio
        self.box_linear_embeddings = Linear(config.hidden_size, width)
        self.box_position_embeddings = Embedding(config.max_position_embeddings, width, padding_idx=config.pad_token_id)
        self.LayerNorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, bbox, positions):
        hidden_states = self.box_linear_embeddings(torch.cat(self.coordinates(bbox), dim=-1))
        return self.LayerNorm(hidden_states + self.box_position_embeddings(positions))


class CoupledSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        self.layout_head_dim = self.head_dim // config.channel_shrink_ratio
        for name in ("query", "key", "value"):
            setattr(self, name, Linear(config.hidden_size, config.hidden_size))
            width = config.hidden_size // config.channel_shrink_ratio
            setattr(self, "layout_" + name, Linear(width, width))
        self.matmul = BMM()
        self.softmax = Softmax()

    def split_heads(self, hidden_states, width):
        return hidden_states.view(*hidden_states.shape[:-1], self.heads, width).transpose(1, 2)

    def forward(self, hidden_states, layout_states):
        query, key, value = [
            self.split_heads(getattr(self, name)(hidden_states), self.head_dim)
            for name in ("query", "key", "value")
        ]
        layout_query, layout_key, layout_value = [
            self.split_heads(getattr(self, "layout_" + name)(layout_states), self.layout_head_dim)
            for name in ("query", "key", "value")
        ]
        text_scores = self.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        layout_scores = self.matmul(layout_query, layout_key.transpose(-1, -2)) / math.sqrt(self.layout_head_dim)
        text_context = self.matmul(self.softmax(text_scores + layout_scores), value)
        layout_context = self.matmul(self.softmax(layout_scores + text_scores), layout_value)
        return (
            text_context.transpose(1, 2).reshape_as(hidden_states),
            layout_context.transpose(1, 2).reshape_as(layout_states),
        )


class CoupledAttention(nn.Module):
    def __init__(self, text_config, layout_config):
        super().__init__()
        self.self = CoupledSelfAttention(text_config)
        self.output = EncoderSelfOutput(text_config)
        self.layout_output = EncoderSelfOutput(layout_config)

    def forward(self, hidden_states, layout_states):
        text_context, layout_context = self.self(hidden_states, layout_states)
        return self.output(text_context, hidden_states), self.layout_output(layout_context, layout_states)


class CoupledLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        layout_config = copy.copy(config)
        layout_config.hidden_size //= config.channel_shrink_ratio
        layout_config.intermediate_size //= config.channel_shrink_ratio
        self.attention = CoupledAttention(config, layout_config)
        self.intermediate = EncoderIntermediate(config)
        self.output = EncoderOutput(config)
        self.layout_intermediate = EncoderIntermediate(layout_config)
        self.layout_output = EncoderOutput(layout_config)

    def forward(self, hidden_states, layout_states):
        hidden_states, layout_states = self.attention(hidden_states, layout_states)
        return (
            self.output(self.intermediate(hidden_states), hidden_states),
            self.layout_output(self.layout_intermediate(layout_states), layout_states),
        )


class LiltModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embeddings = XLMRobertaEmbeddings(config)
        self.layout_embeddings = LayoutEmbeddings(config)
        self.layers = nn.ModuleList(CoupledLayer(config) for _ in range(config.num_hidden_layers))
        self.pooler = Pooler(config)

    def forward(self, input_ids, bbox):
        positions = create_roberta_position_ids_from_input_ids(input_ids, self.padding_idx)
        hidden_states = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        layout_states = self.layout_embeddings(bbox, positions)
        for layer in self.layers:
            hidden_states, layout_states = layer(hidden_states, layout_states)
        return {"last_hidden_state": hidden_states, "pooler_output": self.pooler(hidden_states)}


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or getattr(config, "position_embedding_type", "absolute") != "absolute"
            or config.hidden_size % 6 or config.hidden_size % config.num_attention_heads
            or (config.hidden_size // config.num_attention_heads) % config.channel_shrink_ratio
            or config.intermediate_size % config.channel_shrink_ratio or config.channel_shrink_ratio != 4
            or config.chunk_size_feed_forward or config.output_hidden_states or config.output_attentions):
        raise ValueError("LiLT coverage preserves the default coupled text/layout encoder")
    return LiltModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    del config
    remaining, mapped = dict(state_dict), {}
    for destination in model.state_dict():
        source = destination.replace(".emb.weight", ".weight")
        if source.startswith("layers."):
            source = "encoder.layer." + source.removeprefix("layers.")
        mapped[destination] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped LiLT state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
