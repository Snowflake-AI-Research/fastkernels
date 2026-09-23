"""DPRQuestionEncoder with the checkpoint's unprojected first-token output."""

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L3.bert_model import BertModel

from ..runner import Workload


class DPRQuestionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_model = BertModel(config)
        self.pad_token_id = config.pad_token_id

    def forward(self, input_ids):
        mask = input_ids != self.pad_token_id
        return self.bert_model.forward_with_attention_mask(input_ids, attention_mask=mask)[:, 0]


def build_from_config(config, device, dtype):
    if config.projection_dim != 0 or config.hidden_act != "gelu":
        raise ValueError("DPR coverage preserves the default question encoder without an output projection")
    return DPRQuestionEncoder(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    weights = {}
    for name in model.state_dict():
        source = "question_encoder." + name.replace(".emb.weight", ".weight")
        if ".qkv." in source:
            weights[name] = torch.cat([
                state_dict[source.replace(".qkv.", f".{projection}.")]
                for projection in ("query", "key", "value")
            ])
        else:
            weights[name] = state_dict[source]
    model.load_state_dict(weights)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"pooler_output": model(**inputs)})}
