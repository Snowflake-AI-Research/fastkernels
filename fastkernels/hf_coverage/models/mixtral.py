"""Mixtral causal LM using the existing model and its selected MoE backend."""

import torch

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L4.mixtral import MixtralConfig, MixtralForCausalLM

from .llama import make_workloads


def library_config(config, dtype, num_experts):
    if _tp_size() != 1:
        raise ValueError("The MoE coverage workloads require tensor parallel size 1")
    if config.hidden_act != "silu" or config.tie_word_embeddings:
        raise ValueError("These MoE checkpoints require SiLU experts and an untied head")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("These MoE checkpoints use default RoPE")
    return MixtralConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        num_local_experts=num_experts, num_experts_per_tok=config.num_experts_per_tok,
        dtype=dtype,
    )


def build_from_config(config, device, dtype):
    if config.sliding_window is not None:
        raise ValueError("The selected Mixtral checkpoint disables sliding-window attention")
    return MixtralForCausalLM(library_config(config, dtype, config.num_local_experts)).to(
        device=device, dtype=dtype,
    ).eval()


@torch.no_grad()
def load_moe_decoder_weights(model, state_dict, config, *, qk_norm=False):
    """Copy the pinned HF packed expert tensors and pack its Q/K/V projections."""
    backbone = model.model
    direct = {
        "model.embed_tokens.weight": backbone.embed_tokens.embedding_op.emb.weight,
        "model.norm.weight": backbone.norm.weight,
        "lm_head.weight": model.lm_head.embedding_op.emb.weight,
    }
    packed = {}
    for index, layer in enumerate(backbone.layers):
        prefix = f"model.layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            direct[prefix + name + ".weight"] = getattr(layer, name).weight
        attention = layer.self_attn
        direct[prefix + "self_attn.o_proj.weight"] = attention.o_proj.weight
        if qk_norm:
            for name in ("q_norm", "k_norm"):
                direct[prefix + "self_attn." + name + ".weight"] = getattr(attention, name).weight
        for shard in ("q", "k", "v"):
            heads = config.num_attention_heads if shard == "q" else config.num_key_value_heads
            packed[prefix + f"self_attn.{shard}_proj.weight"] = (
                attention.qkv_proj.weight, shard,
                (heads * model.config.head_dim, config.hidden_size),
            )
        experts = layer.block_sparse_moe
        router = experts.router if qk_norm else experts.gate
        direct[prefix + "mlp.gate.weight"] = router.weight
        direct[prefix + "mlp.experts.gate_up_proj"] = experts.w13
        direct[prefix + "mlp.experts.down_proj"] = experts.w2
    expected = direct.keys() | packed.keys()
    if state_dict.keys() != expected:
        raise KeyError(f"MoE state mismatch: missing={sorted(expected - state_dict.keys())}, extra={sorted(state_dict.keys() - expected)}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"MoE weight shape mismatch for {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard, shape) in packed.items():
        if tuple(state_dict[name].shape) != shape:
            raise ValueError(f"MoE packed projection shape mismatch for {name}")
        parameter.weight_loader(parameter, state_dict[name], shard)


def load_state_dict_into(model, state_dict, config):
    load_moe_decoder_weights(model, state_dict, config)
    for layer in model.model.layers:
        experts = layer.block_sparse_moe
        if experts.w13.dtype == torch.bfloat16:
            # The existing Blackwell backend requires this reversible value
            # permutation into its block layout; it does not quantize weights.
            experts.process_weights_after_loading()
