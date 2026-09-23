"""SmolLM3 default full attention with every fourth layer omitting RoPE."""

from copy import copy

from fastkernels.hf_coverage.models.llama import load_state_dict_into as load_llama, make_workloads as llama_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.attention_bias or config.mlp_bias:
        raise ValueError("SmolLM3 requires bias-free SiLU layers")
    if not config.tie_word_embeddings or config.use_cache or config.use_sliding_window:
        raise ValueError("The selected checkpoint ties embeddings and disables returned cache and windows")
    if config.rope_parameters["rope_type"] != "default":
        raise ValueError("The selected SmolLM3 uses default RoPE")
    if len(config.no_rope_layers) != config.num_hidden_layers or any(t != "full_attention" for t in config.layer_types):
        raise ValueError("SmolLM3 requires a complete full-attention layer schedule")
    model = LlamaForCausalLM(decoder_config(config, dtype))
    for layer, use_rope in zip(model.model.layers, config.no_rope_layers):
        layer.self_attn.nope = not use_rope
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


def make_workloads(model, inputs, config):
    return llama_workloads(model, inputs, config, cached_decode=False)


def load_state_dict_into(model, state_dict, config):
    carrier = copy(config)
    carrier.head_dim = model.config.head_dim
    load_llama(model, state_dict, carrier)
