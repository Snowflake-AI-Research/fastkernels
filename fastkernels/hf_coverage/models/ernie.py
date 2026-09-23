"""ErnieForPreTraining with the pinned task example's ERNIE 1.0 ReLU layers."""

from collections import OrderedDict

import torch
from torch import nn

from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L1.tanh import Tanh

from ..runner import Workload
from .bert import BertForMaskedLM, load_state_dict_into as load_bert_weights


class ErnieForPreTraining(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.bert.encoder.layer:
            layer.intermediate.intermediate_act_fn = ReLU()
        self.lm_head.activation = ReLU()
        self.pooler = nn.Sequential(OrderedDict([
            ("dense", Linear(config.hidden_size, config.hidden_size)), ("activation", Tanh()),
        ]))
        self.seq_relationship = Linear(config.hidden_size, 2)

    def forward(self, input_ids):
        # The default HF call has no attention mask. Keep that boundary rather
        # than creating an all-valid mask, which changes SDPA kernel selection.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        hidden_states = self.bert(input_ids, positions)
        return {
            "prediction_logits": self.lm_head(hidden_states),
            "seq_relationship_logits": self.seq_relationship(self.pooler(hidden_states[:, 0])),
        }


def build_from_config(config, device, dtype):
    if config.hidden_act != "relu" or config.use_task_id or config.is_decoder or config.add_cross_attention:
        raise ValueError("ERNIE coverage preserves its pretraining example's ERNIE 1.0 computation")
    return ErnieForPreTraining(config).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    renamed = {(name.replace("ernie.", "bert.", 1) if name.startswith("ernie.") else name): value
               for name, value in state_dict.items()}
    load_bert_weights(model, renamed, config)
    model.pooler.load_state_dict({name: state_dict["ernie.pooler." + name]
                                 for name in model.pooler.state_dict()})
    model.seq_relationship.load_state_dict({name: state_dict["cls.seq_relationship." + name]
                                           for name in model.seq_relationship.state_dict()})


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: model(**inputs))}
