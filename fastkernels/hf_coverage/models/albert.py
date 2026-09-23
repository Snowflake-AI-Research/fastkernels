"""AlbertForMaskedLM with base-v2's shared encoder and narrow embeddings."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.encoder_embeddings import BertEmbeddings
from fastkernels.tasks.baseline.L3.bert_encoder import BertEncoder

from ..runner import Workload
from .bert import MaskedLMHead


class AlbertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_config = SimpleNamespace(**(dict(config) | {"hidden_size": config.embedding_size}))
        self.embeddings = BertEmbeddings(embedding_config)
        self.embedding_hidden_mapping_in = Linear(config.embedding_size, config.hidden_size)

        shared_config = SimpleNamespace(**(dict(config) | {"num_hidden_layers": 1}))
        self.encoder = BertEncoder(shared_config)
        shared_layer = self.encoder.layer[0]
        shared_layer.intermediate.intermediate_act_fn = GELU(approximate="tanh")
        # These are aliases to one physical layer, invoked once per logical step.
        self.encoder.layer = nn.ModuleList([shared_layer] * config.num_hidden_layers)

        self.lm_head = MaskedLMHead(embedding_config)
        self.lm_head.dense = Linear(config.hidden_size, config.embedding_size)
        self.lm_head.activation = GELU(approximate="tanh")
        if config.tie_word_embeddings:
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.emb.weight

    def forward(self, input_ids):
        positions = self.embeddings.position_ids[:, :input_ids.shape[1]]
        hidden_states = self.embeddings.forward_with_token_type_ids(input_ids, positions)
        hidden_states = self.encoder(self.embedding_hidden_mapping_in(hidden_states))
        return self.lm_head(hidden_states)


def build_from_config(config, device, dtype):
    if (config.num_hidden_layers < 2 or config.num_hidden_groups != 1 or config.inner_group_num != 1
            or config.hidden_act != "gelu_new" or config.embedding_size >= config.hidden_size):
        raise ValueError("ALBERT coverage requires base-v2's shared group and narrow embeddings")
    return AlbertForMaskedLM(config).to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    renames = {
        "attention.output.dense": "attention.dense",
        "attention.output.LayerNorm": "attention.LayerNorm",
        "intermediate.dense": "ffn",
        "output.dense": "ffn_output",
        "output.LayerNorm": "full_layer_layer_norm",
    }
    # named_parameters visits each shared layer/tied embedding only once.
    for name, parameter in model.named_parameters():
        source = name.replace(".emb.weight", ".weight")
        if name.startswith("encoder.layer."):
            suffix = source.split(".", 3)[3]
            for target, reference in renames.items():
                suffix = suffix.replace(target, reference)
            source = "albert.encoder.albert_layer_groups.0.albert_layers.0." + suffix
        elif name.startswith("embedding_hidden_mapping_in."):
            source = "albert.encoder." + source
        elif name.startswith("lm_head."):
            source = source.replace("lm_head.", "predictions.", 1)
        else:
            source = "albert." + source
        if ".attention.self.qkv." in source:
            weight = torch.cat([
                state_dict[source.replace("attention.self.qkv", f"attention.{projection}")]
                for projection in ("query", "key", "value")
            ])
        else:
            weight = state_dict[source]
        parameter.copy_(weight)


def make_workloads(model, inputs, config):
    return {"forward": Workload(run=lambda: {"logits": model(inputs["input_ids"])})}
