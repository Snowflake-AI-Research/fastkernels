"""XLM-RoBERTa-XL omits embedding normalization before its pre-norm blocks."""

from .roberta import make_workloads
from .roberta_prelayernorm import build_pre_norm, load_pre_norm_weights


def build_from_config(config, device, dtype):
    return build_pre_norm(config, device, dtype, normalize_embeddings=False)


def load_state_dict_into(model, state_dict, config):
    load_pre_norm_weights(
        model, state_dict, prefix="roberta",
        attention_norm="attention.self_attn_layer_norm", mlp_norm="LayerNorm",
        final_norm="encoder.LayerNorm",
    )
