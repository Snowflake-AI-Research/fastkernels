"""Qwen2 MoE causal LM using biased-QKV decoder and shared-expert tasks."""

import torch

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM

from .llama import make_workloads


def library_config(config, dtype, *, qkv_bias):
    if _tp_size() != 1:
        raise ValueError("Qwen MoE coverage requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.tie_word_embeddings:
        raise ValueError("These Qwen MoE checkpoints require SiLU and an untied head")
    if config.decoder_sparse_step != 1 or config.mlp_only_layers:
        raise ValueError("These Qwen MoE checkpoints use experts in every decoder layer")
    if config.use_sliding_window or config.sliding_window:
        raise ValueError("These Qwen MoE checkpoints disable sliding windows")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("These Qwen MoE checkpoints use default RoPE")
    return LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0, rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=qkv_bias,
    )


def build_from_config(config, device, dtype):
    if not config.qkv_bias or any(kind != "full_attention" for kind in config.layer_types):
        raise ValueError("Qwen1.5-MoE uses biased QKV and full attention in every layer")
    model = LlamaForCausalLM(library_config(config, dtype, qkv_bias=True))
    for layer in model.model.layers:
        layer.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size, num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            routing="softmax", renormalize=config.norm_topk_prob,
            keep_router_weights_fp32=False,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            shared_expert_gate=True,
        )
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_qwen_moe_weights(model, state_dict, config, *, shared_expert, qk_norm):
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
            width = heads * model.config.head_dim
            packed[prefix + f"self_attn.{shard}_proj.weight"] = (
                attention.qkv_proj.weight, shard, (width, config.hidden_size),
            )
            if attention.qkv_proj.bias is not None:
                packed[prefix + f"self_attn.{shard}_proj.bias"] = (
                    attention.qkv_proj.bias, shard, (width,),
                )
        experts = layer.mlp
        direct[prefix + "mlp.gate.weight"] = experts.gate.weight
        direct[prefix + "mlp.experts.gate_up_proj"] = experts.w13
        direct[prefix + "mlp.experts.down_proj"] = experts.w2
        if shared_expert:
            shared = experts.shared_expert
            direct[prefix + "mlp.shared_expert.down_proj.weight"] = shared.down_proj.weight
            direct[prefix + "mlp.shared_expert_gate.weight"] = experts.shared_expert_gate.weight
            for shard, name in enumerate(("gate_proj", "up_proj")):
                packed[prefix + f"mlp.shared_expert.{name}.weight"] = (
                    shared.gate_up_proj.weight, shard,
                    (config.shared_expert_intermediate_size, config.hidden_size),
                )
    expected = direct.keys() | packed.keys()
    if state_dict.keys() != expected:
        raise KeyError(f"Qwen MoE state mismatch: missing={sorted(expected - state_dict.keys())}, extra={sorted(state_dict.keys() - expected)}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"Qwen MoE weight shape mismatch for {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard, shape) in packed.items():
        if tuple(state_dict[name].shape) != shape:
            raise ValueError(f"Qwen MoE packed projection shape mismatch for {name}")
        parameter.weight_loader(parameter, state_dict[name], shard)


def load_state_dict_into(model, state_dict, config):
    load_qwen_moe_weights(model, state_dict, config, shared_expert=True, qk_norm=False)
    for layer in model.model.layers:
        # The existing Blackwell path permutes BF16 bytes into its kernel layout.
        # Processing neither quantizes the common weights nor changes routing.
        layer.mlp.process_weights_after_loading()
