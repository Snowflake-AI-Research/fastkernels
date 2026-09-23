"""Qwen3 causal LM using the existing per-head Q/K normalization path."""

import torch

from fastkernels.hf_coverage.models.llama import load_state_dict_into as load_llama_weights, make_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("Qwen3 coverage requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.attention_bias or config.tie_word_embeddings:
        raise ValueError("Qwen3-8B requires bias-free SiLU layers and an untied head")
    if config.use_sliding_window or any(kind != "full_attention" for kind in config.layer_types):
        raise ValueError("Qwen3-8B preserves full attention in every layer")
    if config.num_attention_heads != 4 * config.num_key_value_heads:
        raise ValueError("Qwen3-8B preserves 4:1 grouped-query attention")
    if config.head_dim * config.num_attention_heads != config.hidden_size:
        raise ValueError("Qwen3-8B preserves the derived head width")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("Qwen3-8B uses default RoPE")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0, rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    for layer in model.model.layers:
        layer.self_attn.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        layer.self_attn.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    norm_names = {
        f"model.layers.{index}.self_attn.{shard}_norm.weight"
        for index in range(config.num_hidden_layers) for shard in ("q", "k")
    }
    missing = norm_names - set(state_dict)
    if missing:
        raise KeyError(f"Qwen3 state is missing Q/K norm weights: {sorted(missing)}")
    load_llama_weights(model, {k: v for k, v in state_dict.items() if k not in norm_names}, config)
    for index, layer in enumerate(model.model.layers):
        for shard in ("q", "k"):
            source = state_dict[f"model.layers.{index}.self_attn.{shard}_norm.weight"]
            parameter = getattr(layer.self_attn, f"{shard}_norm").weight
            if source.shape != parameter.shape:
                raise ValueError("Qwen3 Q/K norm must have one learned scale per head coordinate")
            parameter.copy_(source)
