"""Seed-OSS causal LM using the existing biased-QKV Llama decoder graph."""

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.qwen2 import load_state_dict_into
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("The Seed-OSS coverage workload requires tensor parallel size 1")
    if (
        config.hidden_act != "silu" or not config.attention_bias
        or config.attention_out_bias or config.mlp_bias or config.tie_word_embeddings
    ):
        raise ValueError("The Seed-OSS checkpoint requires QKV bias, bias-free output/MLP, SiLU, and an untied head")
    if config.num_attention_heads != 10 * config.num_key_value_heads:
        raise ValueError("The Seed-OSS checkpoint preserves 10:1 grouped-query attention")
    if config.num_attention_heads * config.head_dim != 2 * config.hidden_size:
        raise ValueError("The Seed-OSS checkpoint preserves attention width twice the hidden size")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The selected Seed-OSS checkpoint uses default RoPE")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0, rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=True,
    )
    return LlamaForCausalLM(fk_config).to(device=device, dtype=dtype).eval()
