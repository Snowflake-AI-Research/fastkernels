"""Data2VecText masked prediction shares the existing RoBERTa computation."""

from .roberta import build_from_config, make_workloads
from .roberta import load_state_dict_into as load_roberta_weights


def load_state_dict_into(model, state_dict, config):
    weights = {
        name.replace("data2vec_text.", "roberta.", 1): value
        for name, value in state_dict.items()
    }
    load_roberta_weights(model, weights, config)
