"""GLM causal LM with biased QKV and interleaved half-head RoPE."""

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.qwen2 import load_state_dict_into as load_biased_weights
from fastkernels.hf_coverage.models.stablelm import PartialRotary
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("GLM coverage requires tensor parallel size 1")
    if config.hidden_act != "silu" or not config.attention_bias or config.tie_word_embeddings:
        raise ValueError("GLM-4-9B requires SiLU, biased QKV, and an untied head")
    if config.num_attention_heads != 16 * config.num_key_value_heads:
        raise ValueError("GLM-4-9B preserves 16:1 grouped-query attention")
    if config.head_dim * config.num_attention_heads != config.hidden_size:
        raise ValueError("GLM-4-9B preserves the derived head width")
    rope = config.rope_parameters
    if rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 0.5:
        raise ValueError("GLM-4-9B requires default half-head RoPE")
    rotary_dim = config.head_dim // 2
    if config.head_dim % 4:
        raise ValueError("GLM requires an even active rotary width")
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
    model = LlamaForCausalLM(fk_config)
    rotary = PartialRotary(config.head_dim, rotary_dim, config.max_position_embeddings, rope["rope_theta"])
    rotary.rotary = RotaryEmbedding(rotary_dim, config.max_position_embeddings, rope["rope_theta"], is_neox_style=False)
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    unpacked = dict(state_dict)
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}.mlp."
        packed = unpacked.pop(prefix + "gate_up_proj.weight")
        if packed.shape != (2 * config.intermediate_size, config.hidden_size):
            raise ValueError("GLM packed gate/up weight has an incompatible shape")
        for name, source in zip(("gate", "up"), packed.chunk(2, dim=0)):
            target = prefix + name + "_proj.weight"
            if target in unpacked:
                raise KeyError(f"GLM state contains both packed and separate weights: {target}")
            unpacked[target] = source
    load_biased_weights(model, unpacked, config)
