"""EXAONE MoE with local RoPE, global NoPE, and the existing routed decoder."""

from copy import copy

from fastkernels.hf_coverage.models.dots1 import build_from_config as build_dots
from fastkernels.hf_coverage.models.dots1 import load_state_dict_into as load_dots, make_workloads
from fastkernels.tasks.baseline.L2.attention_impl import Attention


def carrier(config):
    if config.mlp_layer_types != ["dense"] + ["sparse"] * (config.num_hidden_layers - 1):
        raise ValueError("Selected EXAONE MoE starts with one dense layer")
    c = copy(config)
    c.layer_types = ["full_attention"] * c.num_hidden_layers
    c.first_k_dense_replace = 1
    c.n_routed_experts, c.n_shared_experts = c.num_experts, c.num_shared_experts
    c.attention_bias = False
    return c


def build_from_config(config, device, dtype):
    model = build_dots(carrier(config), device, dtype)
    for layer, kind in zip(model.model.layers, config.layer_types):
        if kind not in ("full_attention", "sliding_attention"):
            raise ValueError(f"Unknown EXAONE attention type: {kind}")
        local = kind == "sliding_attention"
        a = layer.self_attn.attn
        layer.self_attn.attn = Attention(a.num_heads, a.head_size, a.scale, num_kv_heads=a.num_kv_heads,
                                        sliding_window=config.sliding_window if local else None)
        layer.self_attn.nope = config.sliding_window is not None and not local
        if hasattr(layer.mlp, "gate"):
            bias = layer.mlp.gate.e_score_correction_bias
            bias.data = bias.data.float()
    return model


def load_state_dict_into(model, state_dict, config):
    load_dots(model, state_dict, carrier(config))
