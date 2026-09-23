"""Solar Open routed decoder with existing YaRN and grouped expert operations."""

import torch
from copy import copy

from fastkernels.hf_coverage.models.dots1 import build_from_config as build_dots
from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding


def build_from_config(config, device, dtype):
    c = copy(config)
    c.layer_types = ["full_attention"] * c.num_hidden_layers
    c.first_k_dense_replace = 0
    model = build_dots(c, device, dtype)
    rope = config.rope_parameters
    if rope["rope_type"] != "yarn" or config.partial_rotary_factor != 1.0:
        raise ValueError("Selected Solar Open uses full-head YaRN")
    rotary = YaRNRotaryEmbedding(model.config.head_dim, config.max_position_embeddings,
                                rope["rope_theta"], rope["factor"], rope["original_max_position_embeddings"],
                                beta_fast=rope.get("beta_fast", 32), beta_slow=rope.get("beta_slow", 1)).to(device=device, dtype=dtype)
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.q_norm = layer.self_attn.k_norm = None
        layer.self_attn.rotary_emb = rotary
        bias = layer.mlp.gate.e_score_correction_bias
        bias.data = bias.data.float()
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            direct[p + name + ".weight"] = getattr(layer, name).weight
        direct[p + "self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
        for shard in ("q", "k", "v"):
            packed[p + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard)
        expert = layer.mlp
        direct[p + "mlp.gate.weight"] = expert.gate.weight
        direct[p + "mlp.gate.e_score_correction_bias"] = expert.gate.e_score_correction_bias
        direct[p + "mlp.experts.gate_up_proj"] = expert.w13
        direct[p + "mlp.experts.down_proj"] = expert.w2
        direct[p + "mlp.shared_experts.down_proj.weight"] = expert.shared_expert.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[p + f"mlp.shared_experts.{name}.weight"] = (expert.shared_expert.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"Solar Open state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"Solar Open weight shape/dtype mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
