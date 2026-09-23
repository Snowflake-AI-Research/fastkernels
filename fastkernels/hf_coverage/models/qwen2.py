"""Qwen2 causal LM using the existing biased-QKV Llama model path."""

from __future__ import annotations

import torch

from fastkernels.hf_coverage.models.llama import (
    load_state_dict_into as load_llama_weights,
    make_workloads,
)
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype, *, native_precision=False) -> LlamaForCausalLM:
    if _tp_size() != 1:
        raise ValueError("The Qwen2 coverage workload requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.tie_word_embeddings:
        raise ValueError("The Qwen2 pilot requires SiLU and an untied head")
    if config.use_sliding_window or any(kind != "full_attention" for kind in config.layer_types):
        raise ValueError("The Qwen2-7B pilot preserves its full-attention layers")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The Qwen2-7B pilot requires default RoPE")
    if config.num_attention_heads != 7 * config.num_key_value_heads:
        raise ValueError("The Qwen2-7B pilot preserves 7:1 grouped-query attention")
    head_dim = config.hidden_size // config.num_attention_heads
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=head_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype,
        qkv_bias=True,
    )
    model = LlamaForCausalLM(fk_config)
    if native_precision:
        from .qwen2_precision import NativeRotaryEmbedding, SeparateQKV, configure_language

        # Use the rounding boundaries validated for this full-size case:
        # native normalization/RoPE, dense GQA, and separate Q/K/V projections.
        configure_language(model.model, fk_config)
        model.model.rotary_emb = NativeRotaryEmbedding(
            head_dim, config.max_position_embeddings, rope["rope_theta"],
        )
        sizes = [config.num_attention_heads * head_dim] + [config.num_key_value_heads * head_dim] * 2
        for layer in model.model.layers:
            layer.self_attn.rotary_emb = model.model.rotary_emb
            layer.self_attn.qkv_proj = SeparateQKV(layer.self_attn.qkv_proj, sizes)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config) -> None:
    bias_names = {
        f"model.layers.{index}.self_attn.{shard}_proj.bias"
        for index in range(config.num_hidden_layers)
        for shard in ("q", "k", "v")
    }
    missing = bias_names - set(state_dict)
    if missing:
        raise KeyError(f"Qwen2 state is missing QKV biases: {sorted(missing)}")
    # The remaining state has exactly the Llama carrier's weight contract.
    # Its config contains the derived head dimension used by the packed loader.
    load_llama_weights(
        model, {name: value for name, value in state_dict.items() if name not in bias_names},
        model.config,
    )
    for index, layer in enumerate(model.model.layers):
        bias = layer.self_attn.qkv_proj.bias
        for shard in ("q", "k", "v"):
            source = state_dict[f"model.layers.{index}.self_attn.{shard}_proj.bias"]
            heads = config.num_attention_heads if shard == "q" else config.num_key_value_heads
            if source.shape != (heads * model.config.head_dim,):
                raise ValueError(f"Qwen2 {shard} projection bias has an incompatible shape")
            bias.weight_loader(bias, source, shard)
