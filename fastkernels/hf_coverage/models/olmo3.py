"""OLMo3: OLMo2 postnorm stack with existing YaRN and alternating windows."""

import math
from copy import copy

from fastkernels.hf_coverage.models.olmo2 import build_from_config as build_olmo2, load_state_dict_into
from fastkernels.hf_coverage.models.llama import make_workloads as llama_workloads
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding
from fastkernels.tasks.baseline.L2.attention_impl import Attention


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if rope["rope_type"] != "yarn" or config.use_cache:
        raise ValueError("The selected OLMo3 instruct checkpoint uses YaRN and disables returned cache")
    expected_scale = 1 + 0.1 * math.log(rope["factor"])
    if not math.isclose(rope.get("attention_factor", expected_scale), expected_scale):
        raise ValueError("Selected YaRN scale differs from the existing operation")
    if len(config.layer_types) != config.num_hidden_layers:
        raise ValueError("OLMo3 requires one attention type per layer")
    carrier = copy(config)
    carrier.rope_parameters = {"rope_type": "default", "rope_theta": rope["rope_theta"]}
    model = build_olmo2(carrier, device, dtype)
    rotary = YaRNRotaryEmbedding(
        model.config.head_dim, math.ceil(config.max_position_embeddings / rope["factor"]),
        rope["rope_theta"], rope["factor"], rope["original_max_position_embeddings"],
        beta_fast=rope.get("beta_fast", 32), beta_slow=rope.get("beta_slow", 1),
        truncate=rope.get("truncate", True),
    )
    model.model.rotary_emb = rotary
    for layer, kind in zip(model.model.layers, config.layer_types):
        if kind not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unknown OLMo3 layer type: {kind}")
        layer.self_attn.rotary_emb = rotary
        a = layer.self_attn.attn
        layer.self_attn.attn = Attention(a.num_heads, a.head_size, a.head_size ** -0.5, num_kv_heads=a.num_kv_heads,
            sliding_window=config.sliding_window if kind == "sliding_attention" else None)
    return model.to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config):
    return llama_workloads(model, inputs, config, cached_decode=False)
