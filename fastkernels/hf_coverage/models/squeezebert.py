"""SqueezeBertForMaskedLM with grouped convolutions and its default pooler."""

from collections import OrderedDict

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.conv1d_native import Conv1dNative
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.encoder_attention import EncoderSelfAttention
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_model import BertModel

from ..runner import Workload
from .bert import MaskedLMHead, load_mlm_head


class SqueezeBertEmbeddings(BertEmbeddings):
    def forward_with_token_type_ids(
        self, input_ids, position_ids, token_type_ids=None, inputs_embeds=None,
    ):
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :input_ids.shape[1]].expand_as(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # SqueezeBERT adds position before token type; reversing those additions
        # changes BF16 rounding before the first encoder layer.
        hidden_states = inputs_embeds + self.position_embeddings(position_ids)
        hidden_states = hidden_states + self.token_type_embeddings(token_type_ids)
        return self.LayerNorm(hidden_states)


class GroupedProjection(Conv1dNative):
    def __init__(self, input_size, output_size, groups):
        super().__init__(input_size, output_size, kernel_size=1, groups=groups)

    def forward(self, hidden_states):
        return super().forward(hidden_states.transpose(1, 2)).transpose(1, 2)


class SqueezeBertSelfAttention(EncoderSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        del self.qkv
        self.query = GroupedProjection(config.hidden_size, config.hidden_size, config.q_groups)
        self.key = GroupedProjection(config.hidden_size, config.hidden_size, config.k_groups)
        self.value = GroupedProjection(config.hidden_size, config.hidden_size, config.v_groups)

    def _project_qkv(self, hidden_states):
        return self.query(hidden_states), self.key(hidden_states), self.value(hidden_states)


class SqueezeBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = BertModel(config)
        self.transformer.embeddings = SqueezeBertEmbeddings(config)
        for layer in self.transformer.encoder.layer:
            layer.attention.self = SqueezeBertSelfAttention(config)
            layer.attention.output.dense = GroupedProjection(
                config.hidden_size, config.hidden_size, config.post_attention_groups,
            )
            layer.intermediate.dense = GroupedProjection(
                config.hidden_size, config.intermediate_size, config.intermediate_groups,
            )
            layer.output.dense = GroupedProjection(
                config.intermediate_size, config.hidden_size, config.output_groups,
            )
            # HF's ConvDropoutLayerNorm uses this constructor default directly.
            layer.attention.output.LayerNorm.eps = 1e-12
            layer.output.LayerNorm.eps = 1e-12
        self.pooler = nn.Sequential(OrderedDict([
            ("dense", Linear(config.hidden_size, config.hidden_size)), ("activation", Tanh()),
        ]))
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.transformer.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        hidden_states = self.transformer.forward_with_attention_mask(input_ids)
        pooled_output = self.pooler(hidden_states[:, 0])
        return self.lm_head(hidden_states), pooled_output


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu" or config.embedding_size != config.hidden_size:
        raise ValueError("SqueezeBERT coverage preserves its equal-width grouped-convolution encoder")
    return SqueezeBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    renames = {
        "encoder.layer.": "encoder.layers.", "attention.self.": "attention.",
        "attention.output.dense.": "post_attention.conv1d.",
        "attention.output.LayerNorm.": "post_attention.layernorm.",
        "intermediate.dense.": "intermediate.conv1d.",
        "output.dense.": "output.conv1d.", "output.LayerNorm.": "output.layernorm.",
    }
    for name in model.transformer.state_dict():
        source = name.replace(".emb.weight", ".weight")
        for target, reference in renames.items():
            source = source.replace(target, reference)
        weights[name] = state_dict["transformer." + source]
    model.transformer.load_state_dict(weights)
    model.pooler.load_state_dict({name: state_dict["transformer.pooler." + name]
                                 for name in model.pooler.state_dict()})
    load_mlm_head(model.lm_head, state_dict)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(**inputs)[0]})}
