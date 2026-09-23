"""VisualBERT's documented joint text/visual-feature encoder and CLS pooler."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder
from .mvp import EagerAttention
from ..runner import Workload


class VisualBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.embeddings = nn.ModuleDict({
            "word_embeddings": Embedding(config.vocab_size, width, padding_idx=config.pad_token_id),
            "position_embeddings": Embedding(config.max_position_embeddings, width),
            "token_type_embeddings": Embedding(config.type_vocab_size, width),
            "visual_position_embeddings": Embedding(config.max_position_embeddings, width),
            "visual_token_type_embeddings": Embedding(config.type_vocab_size, width),
            "visual_projection": Linear(config.visual_embedding_dim, width),
            "LayerNorm": LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False),
        })
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            layer.attention.self.attn = EagerAttention(prescale_query=False, divide_scores=True)
        self.pooler = nn.ModuleDict({"dense": Linear(width, width), "activation": Tanh()})

    def forward(self, input_ids, visual_embeds):
        embedding = self.embeddings
        text = embedding["word_embeddings"](input_ids) + embedding["token_type_embeddings"](torch.zeros_like(input_ids))
        text = text + embedding["position_embeddings"](torch.arange(input_ids.shape[1], device=input_ids.device))
        visual_positions = torch.zeros(visual_embeds.shape[:2], device=input_ids.device, dtype=torch.long)
        visual = embedding["visual_projection"](visual_embeds) + embedding["visual_position_embeddings"](visual_positions)
        visual = visual + embedding["visual_token_type_embeddings"](torch.ones_like(visual_positions))
        hidden = self.encoder(embedding["LayerNorm"](torch.cat((text, visual), dim=1)))
        pooled = self.pooler["activation"](self.pooler["dense"](hidden[:, 0]))
        return {"last_hidden_state": hidden, "pooler_output": pooled}


def build_from_config(config, device, dtype):
    if config.bypass_transformer or config.hidden_act != "gelu":
        raise ValueError("The documented checkpoint uses the joint GELU encoder without optional bypass")
    return VisualBertModel(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if ".qkv." in source:
            mapped[name] = torch.cat([remaining.pop(source.replace(".qkv.", f".{part}."))
                                      for part in ("query", "key", "value")])
        else:
            mapped[name] = remaining.pop(source)
    if remaining:
        raise KeyError(f"Unmapped VisualBERT weights: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
