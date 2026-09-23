"""Arcee default forward using an ungated squared-ReLU MLP and YaRN RoPE."""

import math

import torch

from fastkernels.hf_coverage.models.llama import make_workloads as make_llama_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding
from fastkernels.tasks.baseline.L2.vision_mlp import VisionMLP
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("Arcee coverage requires tensor parallel size 1")
    if config.hidden_act != "relu2" or config.attention_bias or config.mlp_bias or config.tie_word_embeddings:
        raise ValueError("AFM-4.5B requires bias-free squared-ReLU layers and an untied head")
    if config.use_cache:
        raise ValueError("AFM-4.5B's default forward disables the returned generation cache")
    if config.num_attention_heads != 5 * config.num_key_value_heads:
        raise ValueError("AFM-4.5B preserves 5:1 grouped-query attention")
    head_dim = config.hidden_size // config.num_attention_heads
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("Arcee requires an integral head width")
    rope = config.rope_parameters
    if (rope["rope_type"] != "yarn" or rope.get("attention_factor") is not None
            or rope.get("mscale_all_dim") or rope.get("partial_rotary_factor", 1.0) != 1.0):
        raise ValueError("AFM-4.5B requires its full-head YaRN magnitude scaling")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=head_dim,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0, rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    # This operation's cache bound is multiplied by factor. Its independent
    # original-context argument preserves the checkpoint's frequency computation.
    rotary = YaRNRotaryEmbedding(
        head_dim, math.ceil(config.max_position_embeddings / rope["factor"]),
        rope["rope_theta"], rope["factor"], rope["original_max_position_embeddings"],
        beta_fast=rope.get("beta_fast", 32), beta_slow=rope.get("beta_slow", 1),
        truncate=rope.get("truncate", True),
    )
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
        layer.mlp = VisionMLP(config.hidden_size, config.intermediate_size, act_fn=SquaredReLU(), bias=False)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for index in range(config.num_hidden_layers):
        expected.update(f"model.layers.{index}." + name for name in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ))
    if set(state_dict) != expected:
        raise KeyError(f"Arcee state mismatch: missing={sorted(expected - set(state_dict))}, extra={sorted(set(state_dict) - expected)}")

    def copy_into(parameter, name):
        source = state_dict[name]
        if source.shape != parameter.shape:
            raise ValueError(f"Arcee weight shape mismatch: {name}")
        parameter.copy_(source)

    copy_into(model.model.embed_tokens.embedding_op.emb.weight, "model.embed_tokens.weight")
    copy_into(model.lm_head.embedding_op.emb.weight, "lm_head.weight")
    copy_into(model.model.norm.weight, "model.norm.weight")
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}."
        copy_into(layer.input_layernorm.weight, prefix + "input_layernorm.weight")
        copy_into(layer.post_attention_layernorm.weight, prefix + "post_attention_layernorm.weight")
        qkv = layer.self_attn.qkv_proj.weight
        for shard in ("q", "k", "v"):
            source = state_dict[prefix + f"self_attn.{shard}_proj.weight"]
            heads = config.num_attention_heads if shard == "q" else config.num_key_value_heads
            if source.shape != (heads * model.config.head_dim, config.hidden_size):
                raise ValueError(f"Arcee {shard} projection has an incompatible shape")
            qkv.weight_loader(qkv, source, shard)
        copy_into(layer.self_attn.o_proj.weight, prefix + "self_attn.o_proj.weight")
        copy_into(layer.mlp.fc1.weight, prefix + "mlp.up_proj.weight")
        copy_into(layer.mlp.fc2.weight, prefix + "mlp.down_proj.weight")


def make_workloads(model, inputs, config):
    # The operation's bounded KV writes are internal overwrite-only workspace;
    # the public task remains one full-sequence forward with use_cache=False.
    return make_llama_workloads(model, inputs, config, cached_decode=False)
