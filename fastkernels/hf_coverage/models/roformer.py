"""RoFormerForMaskedLM with adjacent-pair Q/K rotary position embeddings."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_layer import BertLayer

from ..runner import Workload
from .bert import MaskedLMHead, load_mlm_head
from .rembert import RoundedEncoderAttention


class RoFormerLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention.self.attn = RoundedEncoderAttention()

    def forward(self, hidden_states, positions, cos_sin):
        attention = self.attention.self
        batch, length = hidden_states.shape[:2]
        shape = (batch * length, attention.num_attention_heads, attention.attention_head_size)
        query, key, value = (tensor.reshape(shape) for tensor in attention._project_qkv(hidden_states))
        query, key = RotaryEmbedding.forward_native_interleaved(
            positions, query, key, attention.attention_head_size, cos_sin,
        )
        shape = (batch, length, attention.num_attention_heads, attention.attention_head_size)
        context = attention.attn(query.view(shape), key.view(shape), value.view(shape))
        hidden_states = self.attention.output(context.reshape(batch, length, -1), hidden_states)
        return self.output(self.intermediate(hidden_states), hidden_states)


class RoFormerForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = SimpleNamespace(**(dict(config) | {
            "hidden_size": config.embedding_size, "position_embedding_type": "rotary",
        }))
        self.embeddings = BertEmbeddings(embedding_config)
        del self.embeddings.position_embeddings
        self.embeddings_project = (Linear(config.embedding_size, config.hidden_size)
                                   if config.embedding_size != config.hidden_size else nn.Identity())
        self.embed_positions = nn.Parameter(torch.empty(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads,
        ), requires_grad=False)
        self.layers = nn.ModuleList([RoFormerLayer(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = MaskedLMHead(embedding_config)
        self.lm_head.dense = Linear(config.hidden_size, config.embedding_size)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        batch, length = input_ids.shape
        positions = self.embeddings.position_ids[:, :length]
        hidden_states = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        hidden_states = self.embeddings_project(hidden_states)
        sin, cos = self.embed_positions[:length].chunk(2, dim=-1)
        cos_sin = torch.cat([cos, sin], dim=-1)
        positions = positions.expand(batch, -1).reshape(-1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, cos_sin)
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or config.is_decoder or config.add_cross_attention
            or config.rotary_value or (config.hidden_size // config.num_attention_heads) % 2):
        raise ValueError("RoFormer coverage preserves the checkpoint's Q/K-only rotary encoder")
    return RoFormerForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    for name, parameter in model.named_parameters():
        if name.startswith("lm_head."):
            continue
        source = name.replace(".emb.weight", ".weight")
        if source == "embed_positions":
            source = "encoder.embed_positions.weight"
        elif source.startswith("layers."):
            source = source.replace("layers.", "encoder.layer.", 1)
        source = "roformer." + source
        if ".qkv." in source:
            weight = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weight = state_dict[source]
        parameter.copy_(weight)
    load_mlm_head(model.lm_head, state_dict)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(**inputs)})}
