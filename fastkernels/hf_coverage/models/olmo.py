"""OLMo causal LM using existing Llama attention/MLP and parameter-free LayerNorm."""

import torch

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


class ResidualLayerNorm(LayerNorm):
    """Arrange the decoder residual addition around the existing LayerNorm."""

    def forward(self, hidden_states, residual=None):
        if residual is None:
            return super().forward(hidden_states)
        residual = hidden_states + residual
        return super().forward(residual), residual


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("The OLMo coverage workload requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.attention_bias or config.tie_word_embeddings:
        raise ValueError("The OLMo pilot requires bias-free SiLU layers and an untied head")
    if config.clip_qkv is not None:
        raise ValueError("The selected OLMo checkpoint disables QKV clipping")
    if config.num_attention_heads != config.num_key_value_heads:
        raise ValueError("The OLMo checkpoint preserves multi-head attention")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The OLMo pilot requires default RoPE")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rope_theta=rope["rope_theta"], rope_scaling_factor=1.0,
        rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    for layer in model.model.layers:
        layer.input_layernorm = ResidualLayerNorm(config.hidden_size, elementwise_affine=False)
        layer.post_attention_layernorm = ResidualLayerNorm(config.hidden_size, elementwise_affine=False)
    model.model.norm = ResidualLayerNorm(config.hidden_size, elementwise_affine=False)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_layernorm_decoder_weights(model, state_dict, config, *, affine_norm):
    """Load the shared bias-free decoder graph with optional LayerNorm affine state."""
    backbone = model.model
    direct = {
        "model.embed_tokens.weight": backbone.embed_tokens.embedding_op.emb.weight,
        "lm_head.weight": model.lm_head.embedding_op.emb.weight,
    }
    packed = {}
    if affine_norm:
        direct.update({"model.norm.weight": backbone.norm.weight, "model.norm.bias": backbone.norm.bias})
    for index, layer in enumerate(backbone.layers):
        prefix = f"model.layers.{index}."
        if affine_norm:
            for name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, name)
                direct[prefix + name + ".weight"] = norm.weight
                direct[prefix + name + ".bias"] = norm.bias
        direct[prefix + "self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
        direct[prefix + "mlp.down_proj.weight"] = layer.mlp.down_proj.weight
        for shard in ("q", "k", "v"):
            heads = config.num_attention_heads if shard == "q" else config.num_key_value_heads
            shape = (heads * model.config.head_dim, config.hidden_size)
            packed[prefix + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard, shape)
        for shard, name in enumerate(("gate", "up")):
            packed[prefix + f"mlp.{name}_proj.weight"] = (
                layer.mlp.gate_up_proj.weight, shard, (config.intermediate_size, config.hidden_size),
            )
    expected = direct.keys() | packed.keys()
    if set(state_dict) != expected:
        raise KeyError(f"Decoder state mismatch: missing={sorted(expected - state_dict.keys())}, extra={sorted(state_dict.keys() - expected)}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"Decoder weight shape mismatch for {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard, shape) in packed.items():
        if tuple(state_dict[name].shape) != shape:
            raise ValueError(f"Decoder packed weight shape mismatch for {name}")
        parameter.weight_loader(parameter, state_dict[name], shard)


def load_state_dict_into(model, state_dict, config):
    load_layernorm_decoder_weights(model, state_dict, config, affine_norm=False)
