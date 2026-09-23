"""BertGenerationEncoder using the existing bidirectional BERT stack."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention

from ..runner import Workload
from .distilbert import DistilBertEmbeddings


class BertGenerationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = SimpleNamespace(**(dict(config) | {"dim": config.hidden_size}))
        self.embeddings = DistilBertEmbeddings(embedding_config)
        self.embeddings.LayerNorm.eps = config.layer_norm_eps
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            layer.attention.self.attn = DenseAttention(backend="cudnn")

    def forward(self, input_ids):
        return self.encoder(self.embeddings(input_ids))


def build_from_config(config, device, dtype):
    if config.is_decoder or config.add_cross_attention or config.hidden_act != "gelu":
        raise ValueError("BertGeneration coverage preserves the documented default encoder")
    return BertGenerationEncoder(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if ".qkv." in source:
            weights[name] = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weights[name] = state_dict[source]
    model.load_state_dict(weights)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"last_hidden_state": model(**inputs)})}
