"""ERNIE 4.5 dense/routed layers with FP32 interleaved RoPE and router weights."""

import torch

from fastkernels.hf_coverage.models.dots1 import NormalizedExperts
from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.ernie4_5_rope import FP32RotaryEmbedding
from fastkernels.hf_coverage.patches.grouped_topk_normalization import GroupedTopKNormalization
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


def is_routed(config, index):
    return (config.moe_layer_start_index <= index <= config.moe_layer_end_index
            and (index + 1) % config.moe_layer_interval == 0)


def build_from_config(config, device, dtype):
    if config.use_bias or config.hidden_act != "silu" or not config.tie_word_embeddings:
        raise ValueError("Selected ERNIE MoE uses bias-free SiLU and tied embeddings")
    model = LlamaForCausalLM(decoder_config(config, dtype))
    for i, layer in enumerate(model.model.layers):
        if is_routed(config, i):
            layer.mlp = NormalizedExperts(
                hidden_size=config.hidden_size, num_experts=config.moe_num_experts,
                top_k=config.moe_k, moe_intermediate_size=config.moe_intermediate_size,
                routing="softmax", keep_router_weights_fp32=False,
                shared_expert_intermediate_size=config.moe_intermediate_size * config.moe_num_shared_experts,
                normalizer=GroupedTopKNormalization(scoring_func="softmax", floor=config.moe_norm_min),
            )
    model = model.to(device=device, dtype=dtype).eval()
    rotary = FP32RotaryEmbedding(model.config.head_dim, config.max_position_embeddings,
                               config.rope_parameters["rope_theta"], is_neox_style=False).to(device)
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
        if isinstance(layer.mlp, NormalizedExperts):
            layer.mlp.gate.float()
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
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
        if not is_routed(config, i):
            target, prefix = layer.mlp, p + "mlp."
        else:
            expert = layer.mlp
            direct[p + "mlp.gate.weight"] = expert.gate.weight
            direct[p + "mlp.gate.moe_statics.e_score_correction_bias"] = expert.gate.e_score_correction_bias.view(1, -1)
            direct[p + "mlp.experts.gate_up_proj"] = expert.w13
            direct[p + "mlp.experts.down_proj"] = expert.w2
            target, prefix = expert.shared_expert, p + "mlp.shared_experts."
        direct[prefix + "down_proj.weight"] = target.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[prefix + name + ".weight"] = (target.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"ERNIE MoE state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("ERNIE MoE tied weights disagree")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"ERNIE MoE weight shape/dtype mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
