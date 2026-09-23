"""GLM4-MoE-Lite using the existing compressed MLA implementation."""

from dataclasses import fields

import torch

from fastkernels.hf_coverage.models.dots1 import NormalizedExperts
from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.patches.grouped_topk_normalization import GroupedTopKNormalization
from fastkernels.tasks.baseline.L4.deepseek import DeepSeekV3Config, DeepSeekV3ForCausalLM


def build_from_config(config, device, dtype):
    if config.attention_bias or not config.rope_interleave or config.rope_parameters["rope_type"] != "default":
        raise ValueError("Selected GLM4-MoE-Lite uses bias-free interleaved plain RoPE")
    values = {f.name: getattr(config, f.name) for f in fields(DeepSeekV3Config) if hasattr(config, f.name)}
    values.update(dtype=dtype, rope_theta=config.rope_parameters["rope_theta"],
                  scoring_func="sigmoid", kv_cache_dtype="auto")
    model = DeepSeekV3ForCausalLM(DeepSeekV3Config(**values))
    for i, layer in enumerate(model.model.layers):
        # HF constructs these two norms with their own default epsilon.
        layer.self_attn.q_a_layernorm.eps = 1e-6
        layer.self_attn.kv_a_layernorm.eps = 1e-6
        if i >= config.first_k_dense_replace:
            layer.mlp = NormalizedExperts(
                hidden_size=config.hidden_size, num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                routing="sigmoid", keep_router_weights_fp32=True,
                num_expert_group=config.n_group, topk_group=config.topk_group,
                shared_expert_intermediate_size=config.n_shared_experts * config.moe_intermediate_size,
                normalizer=GroupedTopKNormalization(scoring_func="sigmoid", epsilon=1e-20,
                                                    scale=config.routed_scaling_factor),
            )
    model.to(device=device, dtype=dtype)
    for layer in model.model.layers[config.first_k_dense_replace:]:
        layer.mlp.gate.e_score_correction_bias.data = layer.mlp.gate.e_score_correction_bias.data.float()
    return model.eval()


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
        a = layer.self_attn
        for name in ("q_a_layernorm", "q_b_proj", "kv_a_layernorm", "kv_b_proj", "o_proj"):
            direct[p + "self_attn." + name + ".weight"] = getattr(a, name).weight
        for shard, name in enumerate(("q_a_proj", "kv_a_proj_with_mqa")):
            packed[p + "self_attn." + name + ".weight"] = (a.fused_qkv_a_proj.weight, shard)
        if i < config.first_k_dense_replace:
            target, stem = layer.mlp, p + "mlp."
        else:
            e = layer.mlp
            direct[p + "mlp.gate.weight"] = e.gate.weight
            direct[p + "mlp.gate.e_score_correction_bias"] = e.gate.e_score_correction_bias
            direct[p + "mlp.experts.gate_up_proj"] = e.w13
            direct[p + "mlp.experts.down_proj"] = e.w2
            target, stem = e.shared_expert, p + "mlp.shared_experts."
        direct[stem + "down_proj.weight"] = target.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[stem + name + ".weight"] = (target.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"GLM4-MoE-Lite state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"GLM4-MoE-Lite weight shape mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
    for layer in model.model.layers:
        layer.self_attn.compute_absorbed_weights()
