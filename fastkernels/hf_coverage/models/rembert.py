"""RemBERT's unequal embedding widths around the existing BERT encoder."""

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.linear import BMM, Linear
from fastkernels.tasks.baseline.L1.softmax import Softmax
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from .bert import MaskedLMHead
from .roberta import make_workloads
from ..runner import config_values


class RoundedEncoderAttention(nn.Module):
    """Preserve HF's low-precision score and probability storage boundaries."""

    def __init__(self):
        super().__init__()
        self.matmul = BMM()
        self.softmax = Softmax()

    def forward(self, query, key, value, causal=False, attn_mask=None):
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        scores = self.matmul(query, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        return self.matmul(self.softmax(scores), value).transpose(1, 2)


class RemBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = config_values(config.to_dict())
        embedding_config.hidden_size = config.input_embedding_size
        self.embeddings = BertEmbeddings(embedding_config)
        self.projection = Linear(config.input_embedding_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            layer.attention.self.attn = RoundedEncoderAttention()
        head_config = config_values(config.to_dict())
        head_config.hidden_size = config.output_embedding_size
        self.lm_head = MaskedLMHead(head_config)
        self.lm_head.dense = Linear(config.hidden_size, config.output_embedding_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        positions = self.embeddings.position_ids[:, :input_ids.shape[1]]
        hidden_states = self.embeddings.forward_with_token_type_ids(
            input_ids, positions, token_type_ids=token_type_ids,
        )
        mask = None
        if attention_mask is not None:
            mask = (1 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
        hidden_states = self.encoder.forward_with_attention_mask(
            self.projection(hidden_states), attention_mask=mask,
        )
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or config.is_decoder
            or config.add_cross_attention or config.tie_word_embeddings):
        raise ValueError("RemBERT requires the documented GELU encoder and untied output embeddings")
    return RemBertForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    prefixes = {
        "embeddings": "rembert.embeddings",
        "projection": "rembert.encoder.embedding_hidden_mapping_in",
        "encoder": "rembert.encoder",
        "lm_head": "cls.predictions",
    }
    for name in model.state_dict():
        prefix, rest = name.split(".", 1)
        source = prefixes[prefix] + "." + rest.replace(".emb.weight", ".weight")
        if ".qkv." in source:
            weights[name] = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weights[name] = state_dict[source]
    model.load_state_dict(weights, strict=True)
