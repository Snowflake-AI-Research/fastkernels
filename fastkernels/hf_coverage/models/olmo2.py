"""OLMo2: joint Q/K normalization and post-normalized residual branches."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.bitnet_rms_norm import BitNetRMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM, LlamaModel
from .qwen2_precision import DenseCachedAttention


class NativeAffineRMSNorm(BitNetRMSNorm):
    """Reuse the existing unfused FP32 affine normalization callable."""

    def forward(self, hidden, residual=None):
        if residual is None:
            return self._native_forward(hidden, self.weight, self.eps)
        residual = hidden + residual
        return self._native_forward(residual, self.weight, self.eps), residual


class NativeFP32Rotary(RotaryEmbedding):
    """Reuse native rotation with HF's fixed CPU frequencies and GPU angles."""

    def __init__(self, head_dim, max_positions, theta, device):
        super().__init__(head_dim, max_positions, theta)
        # Position-only constants, prepared once. HF loads inverse frequencies
        # computed on CPU, then evaluates trigonometric functions on the GPU.
        dimensions = torch.arange(0, head_dim, 2, device="cpu", dtype=torch.float32)
        frequency = 1.0 / (theta ** (dimensions / head_dim))
        positions = torch.arange(max_positions, device=device, dtype=torch.float32)
        angles = torch.outer(positions, frequency.to(device))
        self.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)

    def forward(self, positions, query, key):
        q, k = self.forward_native(positions, query.float(), key.float(),
                                    self.head_dim, self.cos_sin_cache)
        return q.to(query.dtype), k.to(key.dtype)


class JointNorm(nn.Module):
    """Restore the complete projected width around the existing RMSNorm."""

    def __init__(self, width, eps):
        super().__init__()
        self.norm = NativeAffineRMSNorm(width, eps)

    def forward(self, x):
        return self.norm(x.reshape(x.shape[0], -1).contiguous()).view_as(x)


class PostNormLayer(nn.Module):
    def __init__(self, layer, eps):
        super().__init__()
        self.self_attn, self.mlp = layer.self_attn, layer.mlp
        width = layer.input_layernorm.hidden_size
        self.post_attention_layernorm = NativeAffineRMSNorm(width, eps)
        self.post_feedforward_layernorm = NativeAffineRMSNorm(width, eps)

    def forward(self, positions, hidden_states, residual=None):
        hidden_states = hidden_states + self.post_attention_layernorm(self.self_attn(positions, hidden_states))
        hidden_states = hidden_states + self.post_feedforward_layernorm(self.mlp(hidden_states))
        return hidden_states, None


class PostNormModel(LlamaModel):
    def forward(self, input_ids, positions, inputs_embeds=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden, _ = layer(positions, hidden)
        return self.norm(hidden)


def decoder_config(config, dtype):
    if _tp_size() != 1:
        raise ValueError("Coverage decoder construction requires tensor parallel size one")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=head_dim,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        rope_theta=config.rope_parameters["rope_theta"], rope_scaling_factor=1.0,
        rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=getattr(config, "attention_bias", False),
    )


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.attention_bias or config.tie_word_embeddings:
        raise ValueError("Selected OLMo2 requires bias-free SiLU and an untied head")
    if config.rope_parameters["rope_type"] != "default":
        raise ValueError("Selected OLMo2 uses default RoPE")
    fk = decoder_config(config, dtype)
    model = LlamaForCausalLM(fk)
    model.model = PostNormModel(fk)
    model.model.layers = nn.ModuleList(PostNormLayer(layer, config.rms_norm_eps) for layer in model.model.layers)
    for layer in model.model.layers:
        layer.self_attn.q_norm = JointNorm(fk.num_attention_heads * fk.head_dim, config.rms_norm_eps)
        layer.self_attn.k_norm = JointNorm(fk.num_key_value_heads * fk.head_dim, config.rms_norm_eps)
        layer.self_attn.attn = DenseCachedAttention(fk.num_attention_heads, fk.num_key_value_heads, fk.head_dim)
    model.model.norm = NativeAffineRMSNorm(config.hidden_size, config.rms_norm_eps)
    model = model.to(device=device, dtype=dtype).eval()
    # Keep the fixed angle table and native rotary products in FP32.
    with torch.device("cpu"):
        rotary = NativeFP32Rotary(fk.head_dim, config.max_position_embeddings,
                                  config.rope_parameters["rope_theta"], device)
    rotary = rotary.to(device=device)
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
            direct[p + name + ".weight"] = getattr(layer, name).weight
        for shard in ("q", "k"):
            direct[p + f"self_attn.{shard}_norm.weight"] = getattr(layer.self_attn, shard + "_norm").norm.weight
        direct[p + "self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
        direct[p + "mlp.down_proj.weight"] = layer.mlp.down_proj.weight
        for shard in ("q", "k", "v"):
            packed[p + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard)
        for shard, name in enumerate(("gate", "up")):
            packed[p + f"mlp.{name}_proj.weight"] = (layer.mlp.gate_up_proj.weight, shard)
    if set(state_dict) != direct.keys() | packed.keys():
        raise KeyError("OLMo2 state keys do not match the complete decoder")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"OLMo2 weight shape mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
