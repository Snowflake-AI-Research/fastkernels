"""Phi-3 causal LM with checkpoint windowing and its native packed state format."""

import torch

from fastkernels.hf_coverage.models.llama import load_state_dict_into as load_llama_weights
from fastkernels.hf_coverage.models.llama import make_workloads as llama_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("The Phi-3 coverage workload requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.tie_word_embeddings:
        raise ValueError("The Phi-3-mini-4k checkpoint requires SiLU and an untied head")
    if config.num_attention_heads != config.num_key_value_heads:
        raise ValueError("The Phi-3-mini-4k checkpoint preserves multi-head attention")
    if config.sliding_window is None or config.sliding_window < 1:
        raise ValueError("The Phi-3-mini-4k checkpoint enables sliding attention")
    rope = config.rope_parameters
    if rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 1.0:
        raise ValueError("The Phi-3-mini-4k checkpoint uses default full-head RoPE")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0, rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    for layer in model.model.layers:
        attention = layer.self_attn.attn
        layer.self_attn.attn = Attention(
            attention.num_heads, attention.head_size, attention.scale,
            num_kv_heads=attention.num_kv_heads, sliding_window=config.sliding_window,
        )
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    # HF already packs Q/K/V and gate/up in the same order. Expose views to
    # the existing strict loader; this conversion runs only during loading.
    unpacked = dict(state_dict)
    head_dim = model.config.head_dim
    widths = [config.num_attention_heads * head_dim] + [config.num_key_value_heads * head_dim] * 2
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        qkv = unpacked.pop(prefix + "self_attn.qkv_proj.weight")
        gate_up = unpacked.pop(prefix + "mlp.gate_up_proj.weight")
        if tuple(qkv.shape) != (sum(widths), config.hidden_size):
            raise ValueError("Phi-3 packed QKV weight has an incompatible shape")
        if tuple(gate_up.shape) != (2 * config.intermediate_size, config.hidden_size):
            raise ValueError("Phi-3 packed gate/up weight has an incompatible shape")
        for name, tensor in zip(("q", "k", "v"), qkv.split(widths, dim=0)):
            key = prefix + f"self_attn.{name}_proj.weight"
            if key in unpacked:
                raise KeyError(f"Unexpected unpacked Phi-3 state key {key}")
            unpacked[key] = tensor
        for name, tensor in zip(("gate", "up"), gate_up.chunk(2, dim=0)):
            key = prefix + f"mlp.{name}_proj.weight"
            if key in unpacked:
                raise KeyError(f"Unexpected unpacked Phi-3 state key {key}")
            unpacked[key] = tensor
    load_llama_weights(model, unpacked, model.config)


def make_workloads(model, inputs, config, *, case=None):
    return llama_workloads(model, inputs, config, case=case,
                           cache_windows=[config.sliding_window] * config.num_hidden_layers)
