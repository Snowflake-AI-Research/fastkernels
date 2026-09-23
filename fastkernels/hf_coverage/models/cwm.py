"""CWM causal LM with Llama-3 RoPE and its full/sliding layer pattern."""

from fastkernels.hf_coverage.models.llama import load_state_dict_into, make_workloads as llama_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("CWM coverage requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.mlp_bias or config.tie_word_embeddings:
        raise ValueError("CWM defaults require bias-free SiLU layers and an untied head")
    if config.num_attention_heads != 6 * config.num_key_value_heads:
        raise ValueError("CWM defaults preserve 6:1 grouped-query attention")
    if config.head_dim * config.num_attention_heads != config.hidden_size:
        raise ValueError("CWM defaults preserve the derived head width")
    pattern = ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
    if config.num_hidden_layers % 4 or config.layer_types != pattern * (config.num_hidden_layers // 4):
        raise ValueError("CWM defaults preserve one full and three sliding layers per group")
    if config.sliding_window is None or config.sliding_window < 1:
        raise ValueError("CWM defaults require an enabled sliding window")
    rope = config.rope_parameters
    if rope["rope_type"] != "llama3":
        raise ValueError("CWM defaults require Llama-3 frequency scaling")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=rope["factor"], rope_low_freq_factor=rope["low_freq_factor"],
        rope_high_freq_factor=rope["high_freq_factor"],
        rope_original_max_position_embeddings=rope["original_max_position_embeddings"],
        dtype=dtype, qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    for layer, kind in zip(model.model.layers, config.layer_types):
        if kind == "sliding_attention":
            attention = layer.self_attn.attn
            layer.self_attn.attn = Attention(
                attention.num_heads, attention.head_size, attention.scale,
                num_kv_heads=attention.num_kv_heads, sliding_window=config.sliding_window,
            )
    return model.to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config, *, case=None):
    windows = [config.sliding_window if kind == "sliding_attention" else None
               for kind in config.layer_types]
    return llama_workloads(model, inputs, config, case=case, cache_windows=windows)
