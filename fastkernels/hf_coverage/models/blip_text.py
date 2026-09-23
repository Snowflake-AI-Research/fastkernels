"""BLIP's default bidirectional text model, including its default pooler."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import Workload
from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from .mvp import EagerAttention


class BlipTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            "word_embeddings": Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            "position_embeddings": Embedding(config.max_position_embeddings, config.hidden_size),
            "LayerNorm": LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False),
        })
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            layer.attention.self.attn = EagerAttention(prescale_query=False, divide_scores=True)
        self.pooler = nn.ModuleDict({"dense": Linear(config.hidden_size, config.hidden_size), "activation": Tanh()})

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.embeddings["LayerNorm"](
            self.embeddings["word_embeddings"](ids) + self.embeddings["position_embeddings"](positions))
        hidden = self.encoder(hidden)
        pooled = self.pooler["activation"](self.pooler["dense"](hidden[:, 0]))
        return {"last_hidden_state": hidden, "pooler_output": pooled}


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu":
        raise ValueError("BLIP text default model uses exact GELU")
    return BlipTextModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights, consumed = {}, set()
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        names = ([source.replace(".qkv.", f".{projection}.") for projection in ("query", "key", "value")]
                 if ".qkv." in source else [source])
        weights[name] = torch.cat([state_dict[key] for key in names]) if len(names) == 3 else state_dict[source]
        consumed.update(names)
    # Default BlipTextModel(input_ids) has is_decoder=False at call time and
    # no encoder_hidden_states. Constructed cross-attention weights are inactive.
    inactive = {name for name in state_dict if ".crossattention." in name}
    if consumed | inactive != set(state_dict):
        raise ValueError(f"BLIP text unmapped state: {sorted(set(state_dict) - consumed - inactive)}")
    model.load_state_dict(weights)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(inputs["input_ids"]))}
