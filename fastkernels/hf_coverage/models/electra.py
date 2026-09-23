"""ElectraForMaskedLM with the small generator's embedding projection."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from ..runner import Workload
from .bert import MaskedLMHead


class ElectraForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = SimpleNamespace(**(dict(config) | {"hidden_size": config.embedding_size}))
        self.embeddings = BertEmbeddings(embedding_config)
        self.embeddings_project = Linear(config.embedding_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.lm_head = MaskedLMHead(embedding_config)
        self.lm_head.dense = Linear(config.hidden_size, config.embedding_size)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        positions = self.embeddings.position_ids[:, :input_ids.shape[1]]
        hidden_states = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        hidden_states = self.encoder(self.embeddings_project(hidden_states))
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or config.is_decoder or config.add_cross_attention
            or config.embedding_size == config.hidden_size
            or getattr(config, "position_embedding_type", "absolute") != "absolute"):
        raise ValueError("ELECTRA coverage requires the small generator's encoder and embedding projection")
    return ElectraForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if name.startswith("lm_head."):
            source = source.replace("lm_head.dense", "generator_predictions.dense")
            source = source.replace("lm_head.LayerNorm", "generator_predictions.LayerNorm")
            source = source.replace("lm_head.decoder", "generator_lm_head")
        else:
            source = "electra." + source
        if ".qkv." in source:
            weights[name] = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weights[name] = state_dict[source]
    model.load_state_dict(weights)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(inputs["input_ids"])})}
