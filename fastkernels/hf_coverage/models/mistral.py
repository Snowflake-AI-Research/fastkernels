"""Mistral causal LM using the existing Llama model and windowed Attention."""

from __future__ import annotations

from fastkernels.hf_coverage.models.llama import load_state_dict_into, make_workloads as llama_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype) -> LlamaForCausalLM:
    if _tp_size() != 1:
        raise ValueError("The Mistral coverage workload requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.tie_word_embeddings:
        raise ValueError("The Mistral pilot requires SiLU and an untied head")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The Mistral-7B-v0.1 pilot requires default RoPE")
    if config.num_attention_heads != 4 * config.num_key_value_heads:
        raise ValueError("The Mistral-7B-v0.1 pilot preserves 4:1 grouped-query attention")
    if config.head_dim != config.hidden_size // config.num_attention_heads:
        raise ValueError("The Mistral pilot preserves the derived head dimension")
    if config.sliding_window is None or config.sliding_window < 1:
        raise ValueError("The Mistral-7B-v0.1 pilot requires its uniform sliding window")

    fk_config = LlamaConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype,
        qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    for layer in model.model.layers:
        attention = layer.self_attn.attn
        # L4/L3 do not expose the window argument. The existing L2 operation
        # does; construct it with that argument for every Mistral layer.
        layer.self_attn.attn = Attention(
            attention.num_heads,
            attention.head_size,
            attention.scale,
            num_kv_heads=attention.num_kv_heads,
            sliding_window=config.sliding_window,
        )
    return model.to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config, *, case=None):
    return llama_workloads(model, inputs, config, case=case,
                           cache_windows=[config.sliding_window] * config.num_hidden_layers)
