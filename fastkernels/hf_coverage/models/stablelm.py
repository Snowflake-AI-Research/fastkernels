"""StableLM causal LM with affine LayerNorm and a leading partial RoPE slice."""

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo import ResidualLayerNorm, load_layernorm_decoder_weights
from fastkernels.hf_coverage.models.qwen2_precision import DenseCachedAttention
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


class PartialRotary(nn.Module):
    """Apply the existing full-slice RoPE operation to each head's leading slice."""

    def __init__(self, head_dim, rotary_dim, max_positions, theta):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rotary = RotaryEmbedding(rotary_dim, max_positions, theta)

    def _rotate(self, positions, query, key):
        return self.rotary(positions, query, key)

    def forward(self, positions, query, key):
        count = query.shape[0]
        query_heads = query.view(count, -1, self.head_dim)
        key_heads = key.view(count, -1, self.head_dim)
        q_rotated, k_rotated = self._rotate(
            positions,
            query_heads[..., :self.rotary_dim].reshape(count, -1).contiguous(),
            key_heads[..., :self.rotary_dim].reshape(count, -1).contiguous(),
        )
        query = torch.cat((q_rotated.view(count, -1, self.rotary_dim), query_heads[..., self.rotary_dim:]), dim=-1)
        key = torch.cat((k_rotated.view(count, -1, self.rotary_dim), key_heads[..., self.rotary_dim:]), dim=-1)
        return query.reshape(count, -1), key.reshape(count, -1)


class NativePartialRotary(PartialRotary):
    """Select the existing rotary operation with separately rounded products."""

    def _rotate(self, positions, query, key):
        return self.rotary.forward_native(
            positions, query, key, self.rotary_dim,
            self.rotary.cos_sin_cache.to(query.dtype),
        )


class SeparateGateUp(nn.Module):
    """Keep packed weights while preserving the two native linear call shapes."""

    def __init__(self, projection):
        super().__init__()
        self.weight = projection.weight
        self.matmul = Matmul()

    def forward(self, hidden):
        return torch.cat([self.matmul(hidden, weight) for weight in self.weight.chunk(2)], dim=-1)


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("The StableLM coverage workload requires tensor parallel size 1")
    if (
        config.hidden_act != "silu" or config.use_qkv_bias or config.qk_layernorm
        or config.use_parallel_residual or config.tie_word_embeddings
    ):
        raise ValueError("The selected StableLM checkpoint requires bias-free SiLU, sequential residuals, and no QK norm")
    if config.num_attention_heads != config.num_key_value_heads:
        raise ValueError("The StableLM checkpoint preserves multi-head attention")
    rope = config.rope_parameters
    if rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 0.25:
        raise ValueError("The selected StableLM checkpoint requires default quarter-head RoPE")
    head_dim = config.hidden_size // config.num_attention_heads
    rotary_dim = int(head_dim * rope["partial_rotary_factor"])
    if config.hidden_size % config.num_attention_heads or rotary_dim < 2 or rotary_dim % 2:
        raise ValueError("StableLM dimensions require integral heads and an even active rotary width")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=head_dim,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rope_theta=rope["rope_theta"], rope_scaling_factor=1.0,
        rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    with torch.device("cpu"):
        rotary = NativePartialRotary(head_dim, rotary_dim, config.max_position_embeddings, rope["rope_theta"])
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
        # The default 80-wide heads need the existing padded NHD cache store;
        # the HND store assumes a power-of-two head width.
        layer.self_attn.attn = DenseCachedAttention(
            config.num_attention_heads, config.num_key_value_heads, head_dim,
        )
        layer.input_layernorm = ResidualLayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        layer.post_attention_layernorm = ResidualLayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
        layer.mlp.gate_up_proj = SeparateGateUp(layer.mlp.gate_up_proj)
    model.model.norm = ResidualLayerNorm(config.hidden_size, config.layer_norm_eps, promote_fp32=False)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    load_layernorm_decoder_weights(model, state_dict, config, affine_norm=True)
