"""BertForMaskedLM using the existing FastKernels BERT encoder."""

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L3.bert_model import BertModel

from ..runner import Workload


class MaskedLMHead(nn.Module):
    """The shared BERT and legacy DeBERTa masked-token prediction head."""

    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.activation = GELU()
        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False,
        )
        self.decoder = Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states):
        return self.decoder(self.LayerNorm(self.activation(self.dense(hidden_states))))


def load_mlm_head(head, state_dict):
    weights = {}
    for name in head.state_dict():
        prefix = "cls.predictions." if name.startswith("decoder.") else "cls.predictions.transform."
        weights[name] = state_dict[prefix + name]
    head.load_state_dict(weights)


class BertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        # Keep cuDNN available despite dependencies that disable global SDPA
        # selection. This is the existing operation's explicit backend.
        for layer in self.bert.encoder.layer:
            layer.attention.self.attn = DenseAttention(backend="cudnn")
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.bert.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        # The declared workload has no padding mask. An unnecessary all-true
        # mask changes SDPA kernel selection and its rounding compared with HF.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        hidden_states = self.bert(input_ids, positions)
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or config.is_decoder
            or config.add_cross_attention
            or getattr(config, "position_embedding_type", "absolute") != "absolute"):
        raise ValueError("BERT coverage requires the bert-base-uncased encoder computation")
    return BertForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    for name in model.bert.state_dict():
        source = "bert." + name.replace(".emb.weight", ".weight")
        if ".qkv." in source:
            weights[name] = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weights[name] = state_dict[source]
    model.bert.load_state_dict(weights)
    load_mlm_head(model.lm_head, state_dict)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(inputs["input_ids"])})}
