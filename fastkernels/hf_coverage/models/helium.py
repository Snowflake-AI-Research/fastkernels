"""Helium causal LM: existing Llama stack with interleaved RoPE and FP32 affine norm."""

from fastkernels.hf_coverage.models.llama import (
    build_from_config as build_llama,
    load_state_dict_into,
    make_workloads,
)
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from .glm4 import NativeInterleavedRotary
from .olmo2 import NativeAffineRMSNorm
from .qwen2_precision import DenseCachedAttention
from .stablelm import SeparateGateUp


def build_from_config(config, device, dtype):
    model = build_llama(config, device, dtype)
    rope = RotaryEmbedding(
        config.head_dim, config.max_position_embeddings,
        config.rope_parameters["rope_theta"], is_neox_style=False,
    )
    rope = NativeInterleavedRotary(rope)
    model.model.rotary_emb = rope
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rope
        layer.self_attn.attn = DenseCachedAttention(
            config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        )
        layer.input_layernorm = NativeAffineRMSNorm(config.hidden_size, config.rms_norm_eps)
        layer.post_attention_layernorm = NativeAffineRMSNorm(config.hidden_size, config.rms_norm_eps)
        layer.mlp.gate_up_proj = SeparateGateUp(layer.mlp.gate_up_proj)
    model.model.norm = NativeAffineRMSNorm(config.hidden_size, config.rms_norm_eps)
    return model.to(device=device, dtype=dtype).eval()
