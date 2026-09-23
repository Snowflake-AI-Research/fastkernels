"""DistilBertForMaskedLM using the existing FastKernels BERT encoder stack."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from ..runner import Workload
from .bert import MaskedLMHead


class DistilBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.dim)
        self.LayerNorm = LayerNorm(config.dim, eps=1e-12, promote_fp32=False)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings)[None], persistent=False,
        )

    def forward(self, input_ids):
        positions = self.position_ids[:, :input_ids.shape[1]]
        return self.LayerNorm(self.word_embeddings(input_ids) + self.position_embeddings(positions))


class DistilBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = SimpleNamespace(
            hidden_size=config.dim, num_attention_heads=config.n_heads,
            num_hidden_layers=config.n_layers, intermediate_size=config.hidden_dim,
            layer_norm_eps=1e-12, vocab_size=config.vocab_size,
        )
        self.embeddings = DistilBertEmbeddings(config)
        self.encoder = BertEncoder(encoder_config)
        self.lm_head = MaskedLMHead(encoder_config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        return self.lm_head(self.encoder(self.embeddings(input_ids)))


def build_from_config(config, device, dtype):
    if config.sinusoidal_pos_embds or config.activation != "gelu":
        raise ValueError("DistilBERT coverage requires the base checkpoint computation")
    return DistilBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    renames = {
        "attention.output.dense": "attention.out_lin",
        "attention.output.LayerNorm": "sa_layer_norm",
        "intermediate.dense": "ffn.lin1",
        "output.dense": "ffn.lin2",
        "output.LayerNorm": "output_layer_norm",
        "lm_head.dense": "vocab_transform",
        "lm_head.LayerNorm": "vocab_layer_norm",
        "lm_head.decoder": "vocab_projector",
    }
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if name.startswith("embeddings."):
            source = "distilbert." + source
        elif name.startswith("encoder."):
            source = source.replace("encoder.", "distilbert.transformer.", 1)
        if ".attention.self.qkv." in source:
            weights[name] = torch.cat([
                state_dict[source.replace("attention.self.qkv", f"attention.{projection}_lin")]
                for projection in ("q", "k", "v")
            ])
        else:
            for target, reference in renames.items():
                source = source.replace(target, reference)
            weights[name] = state_dict[source]
    model.load_state_dict(weights)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(inputs["input_ids"])})}
