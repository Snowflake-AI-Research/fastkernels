"""SplinterModel's bare bidirectional encoder using the existing BERT stack."""

import torch

from fastkernels.tasks.baseline.L3.bert_model import BertModel

from ..runner import Workload
from .bert_generation import load_state_dict_into
from .rembert import RoundedEncoderAttention


class SplinterModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.encoder.layer:
            layer.attention.self.attn = RoundedEncoderAttention()

    def _prepare_attention_mask(self, attention_mask, device):
        if attention_mask is None:
            return None
        dtype = self.embeddings.word_embeddings.emb.weight.dtype
        mask = attention_mask[:, None, None, :].to(device=device, dtype=dtype)
        return (1 - mask) * torch.finfo(dtype).min


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu":
        raise ValueError("Splinter coverage preserves the default bare encoder")
    return SplinterModel(config).to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {
        "last_hidden_state": model.forward_with_attention_mask(**inputs),
    })}
