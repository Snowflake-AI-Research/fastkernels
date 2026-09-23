"""GLM-4's additional post-attention and post-MLP normalization."""

from torch import nn

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from .glm import build_from_config as build_glm, load_state_dict_into as load_glm
from .exaone4 import PostNormBackbone
from .llama import make_workloads


class NativeInterleavedRotary(nn.Module):
    """Select the unchanged native operation's BF16 product-rounding boundary."""

    def __init__(self, rotary):
        super().__init__()
        self.rotary = rotary

    def forward(self, positions, query, key):
        return RotaryEmbedding.forward_native_interleaved(
            positions, query, key, self.rotary.head_dim,
            self.rotary.cos_sin_cache.to(query.dtype),
        )


class Glm4Layer(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.self_attn = layer.self_attn
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.post_self_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden, residual=None):
        if residual is not None:
            hidden = hidden + residual
        attention = self.self_attn(positions, self.input_layernorm(hidden))
        hidden = hidden + self.post_self_attn_layernorm(attention)
        hidden = hidden + self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(hidden)))
        return hidden, None


def build_from_config(config, device, dtype):
    model = build_glm(config, device, dtype)
    model.model.rotary_emb.rotary = NativeInterleavedRotary(model.model.rotary_emb.rotary)
    model.model.layers = nn.ModuleList([Glm4Layer(layer, config) for layer in model.model.layers])
    model.model = PostNormBackbone(model.model)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    for index, layer in enumerate(model.model.layers):
        for name in ("post_self_attn_layernorm", "post_mlp_layernorm"):
            value = remaining.pop(f"model.layers.{index}.{name}.weight")
            parameter = getattr(layer, name).weight
            if value.shape != parameter.shape:
                raise ValueError(f"GLM-4 {name} shape mismatch")
            parameter.data.copy_(value)
    load_glm(model, remaining, config)
