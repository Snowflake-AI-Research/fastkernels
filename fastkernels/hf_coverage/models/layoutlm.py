"""LayoutLM masked language modeling with learned document-box embeddings."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.embedding import Embedding
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.clip_attention import CLIPAttention
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from ..runner import Workload
from .bert import MaskedLMHead


class BoxEmbeddings(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.x_position_embeddings = Embedding(config.max_2d_position_embeddings, width)
        self.y_position_embeddings = Embedding(config.max_2d_position_embeddings, width)
        self.h_position_embeddings = Embedding(config.max_2d_position_embeddings, width)
        self.w_position_embeddings = Embedding(config.max_2d_position_embeddings, width)

    def coordinates(self, bbox):
        # Widths and heights are integer input metadata, not hidden activations.
        return (
            self.x_position_embeddings(bbox[..., 0]),
            self.y_position_embeddings(bbox[..., 1]),
            self.x_position_embeddings(bbox[..., 2]),
            self.y_position_embeddings(bbox[..., 3]),
            self.h_position_embeddings(bbox[..., 3] - bbox[..., 1]),
            self.w_position_embeddings(bbox[..., 2] - bbox[..., 0]),
        )


class LayoutEmbeddings(BoxEmbeddings):
    def __init__(self, config):
        super().__init__(config, config.hidden_size)
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, input_ids, bbox):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden_states = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        for embedding in self.coordinates(bbox):
            hidden_states = hidden_states + embedding
        hidden_states = hidden_states + self.token_type_embeddings(torch.zeros_like(input_ids))
        return self.LayerNorm(hidden_states)


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.activation = Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))


class LayoutAttention(nn.Module):
    """Reuse CLIP's explicit attention to retain HF's BF16 score rounding."""

    def __init__(self, config):
        super().__init__()
        self.core = CLIPAttention(config)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, hidden_states):
        return self.LayerNorm(self.core(hidden_states) + hidden_states)


class LayoutLMForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = LayoutEmbeddings(config)
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            layer.attention = LayoutAttention(config)
        self.pooler = Pooler(config)
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids, bbox):
        hidden_states = self.encoder(self.embeddings(input_ids, bbox))
        # The reference executes its default pooler before its MLM head,
        # although MaskedLMOutput does not expose the pooled tensor.
        pooled_output = self.pooler(hidden_states)
        return {"logits": self.lm_head(hidden_states)}


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or getattr(config, "position_embedding_type", "absolute") != "absolute"
            or config.chunk_size_feed_forward or config.output_hidden_states or config.output_attentions):
        raise ValueError("LayoutLM coverage preserves the default document masked-LM computation")
    return LayoutLMForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining, mapped = dict(state_dict), {}
    for destination in model.state_dict():
        if destination.startswith("lm_head."):
            name = destination.removeprefix("lm_head.")
            source = ("cls.predictions." if name.startswith("decoder.") else "cls.predictions.transform.") + name
        else:
            source = "layoutlm." + destination.replace(".emb.weight", ".weight")
            for projection, native in (("q", "query"), ("k", "key"), ("v", "value")):
                source = source.replace(f".attention.core.{projection}_proj.", f".attention.self.{native}.")
            source = source.replace(".attention.core.out_proj.", ".attention.output.dense.")
            source = source.replace(".attention.LayerNorm.", ".attention.output.LayerNorm.")
        mapped[destination] = remaining.pop(source)
    if not torch.equal(remaining.pop("cls.predictions.bias"), mapped["lm_head.decoder.bias"]):
        raise ValueError("HF masked-LM bias aliases disagree")
    if config.tie_word_embeddings and not torch.equal(
        mapped["lm_head.decoder.weight"], mapped["embeddings.word_embeddings.emb.weight"]
    ):
        raise ValueError("HF tied word embeddings disagree")
    if remaining:
        raise KeyError(f"Unmapped LayoutLM state entries: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    del config
    if set(inputs) != {"input_ids", "bbox"}:
        raise ValueError("Default document inference expects input_ids and bbox")
    return {"forward": Workload(run=lambda: model(**inputs))}
