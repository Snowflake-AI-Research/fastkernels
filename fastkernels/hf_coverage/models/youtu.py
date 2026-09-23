"""Youtu's dense decoder with the existing compressed latent attention."""

import torch

from . import glm4_moe_lite
from .llama import make_workloads
from ..runner import config_values


def dense_config(config):
    values = config.to_dict()
    values["first_k_dense_replace"] = config.num_hidden_layers
    return config_values(values)


def build_from_config(config, device, dtype):
    model = glm4_moe_lite.build_from_config(dense_config(config), device, dtype)
    if config.tie_word_embeddings:
        model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model


def load_state_dict_into(model, state_dict, config):
    if config.tie_word_embeddings and not torch.equal(state_dict["lm_head.weight"],
                                                       state_dict["model.embed_tokens.weight"]):
        raise ValueError("Youtu tied head and embedding weights differ")
    glm4_moe_lite.load_state_dict_into(model, state_dict, dense_config(config))
