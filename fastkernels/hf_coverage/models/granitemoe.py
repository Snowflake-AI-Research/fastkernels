"""Granite MoE decoder using existing scaled decoder and routed-expert operations."""

import torch

from fastkernels.hf_coverage.models.granite import GraniteForCausalLM, make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.attention_bias or not config.tie_word_embeddings:
        raise ValueError("Selected Granite MoE requires bias-free SiLU and tied embeddings")
    if config.rope_parameters["rope_type"] != "default":
        raise ValueError("Selected Granite MoE uses default RoPE")
    model = GraniteForCausalLM(decoder_config(config, dtype), config)
    for layer in model.model.layers:
        layer.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size, num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok, moe_intermediate_size=config.intermediate_size,
            routing="softmax", renormalize=True, keep_router_weights_fp32=False,
            shared_expert_intermediate_size=getattr(config, "shared_intermediate_size", 0),
        )
    return model.to(device=device, dtype=dtype).eval()


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
        experts = layer.mlp
        direct[p + "block_sparse_moe.router.layer.weight"] = experts.gate.weight
        direct[p + "block_sparse_moe.input_linear.weight"] = experts.w13
        direct[p + "block_sparse_moe.output_linear.weight"] = experts.w2
        if experts.has_shared_expert:
            direct[p + "shared_mlp.input_linear.weight"] = experts.shared_expert.gate_up_proj.weight
            direct[p + "shared_mlp.output_linear.weight"] = experts.shared_expert.down_proj.weight
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"Granite MoE state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("Granite MoE tied weights disagree")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"Granite MoE weight shape mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
    for layer in model.model.layers:
        if layer.mlp.w13.dtype == torch.bfloat16:
            layer.mlp.process_weights_after_loading()
