"""Flex OLMo combines post-normalized decoder blocks with unrenormalized experts."""

import torch

from fastkernels.tasks.baseline.L2.jamba_moe import JambaMoE

from .llama import make_workloads
from .olmo2 import build_from_config as build_post_norm_decoder


def build_from_config(config, device, dtype):
    if config.norm_topk_prob or config.output_router_logits:
        raise ValueError("The documented checkpoint uses raw routing weights and omits router outputs")
    model = build_post_norm_decoder(config, device, dtype)
    for layer in model.model.layers:
        layer.mlp = JambaMoE(config.hidden_size, config.intermediate_size,
                            config.num_experts, config.num_experts_per_tok).to(device=device, dtype=dtype)
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}."
        for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
            direct[prefix + name + ".weight"] = getattr(layer, name).weight
        for name in ("q", "k"):
            direct[prefix + f"self_attn.{name}_norm.weight"] = getattr(layer.self_attn, name + "_norm").norm.weight
        direct[prefix + "self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
        for shard in ("q", "k", "v"):
            packed[prefix + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard)
        direct[prefix + "mlp.gate.weight"] = layer.mlp.router.weight
        direct[prefix + "mlp.experts.gate_up_proj"] = layer.mlp.w13
        direct[prefix + "mlp.experts.down_proj"] = layer.mlp.w2
    expected = direct.keys() | packed.keys()
    if state_dict.keys() != expected:
        raise KeyError(f"Flex OLMo mapping: missing={expected - state_dict.keys()}, extra={state_dict.keys() - expected}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"Flex OLMo weight shape mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
