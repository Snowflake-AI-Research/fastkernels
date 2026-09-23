"""RoBERTa-family masked language models using the existing XLM-RoBERTa stack."""

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention
from fastkernels.tasks.baseline.L2.encoder_embeddings import create_roberta_position_ids_from_input_ids
from fastkernels.tasks.baseline.L3.xlm_roberta_model import XLMRobertaModel

from .bert import MaskedLMHead
from ..runner import Workload


class RobertaForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = XLMRobertaModel(config)
        for layer in self.roberta.encoder.layer:
            layer.attention.self.attn = DenseAttention(backend="cudnn")
        self.lm_head = MaskedLMHead(config)
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            # HF omits the mask when none is supplied. The library wrapper adds
            # an all-true mask, which changes SDPA selection and BF16 rounding.
            positions = create_roberta_position_ids_from_input_ids(
                input_ids, self.roberta.config.pad_token_id,
            )
            hidden_states = self.roberta.embeddings.forward_with_token_type_ids(input_ids, positions)
            hidden_states = self.roberta.encoder(hidden_states)
        else:
            hidden_states = self.roberta.forward_with_attention_mask(
                input_ids, attention_mask=attention_mask,
            )
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (config.hidden_act != "gelu" or config.is_decoder
            or config.add_cross_attention
            or getattr(config, "position_embedding_type", "absolute") != "absolute"):
        raise ValueError("RoBERTa coverage requires the documented bidirectional GELU encoder")
    return RobertaForMaskedLM(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    for name in model.state_dict():
        source = name.replace(".emb.weight", ".weight")
        if source.startswith("lm_head."):
            source = source.replace(".LayerNorm.", ".layer_norm.")
        if ".qkv." in source:
            weights[name] = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weights[name] = state_dict[source]
    model.load_state_dict(weights)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(**inputs)})}
