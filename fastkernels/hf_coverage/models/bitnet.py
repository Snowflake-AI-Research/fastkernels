"""BitNet with existing quantization and linear operations in HF's rounding order."""

import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.bitnet_linear import _fake_quant_act_bf16
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L4.bitnet import BitNetConfig, BitNetForCausalLM

from ..patches.product_gate import ProductGate
from .llama import make_workloads


class NativeRotary(nn.Module):
    """Reuse the existing split-half operation with separate BF16 products."""

    def __init__(self, rotary):
        super().__init__()
        self.rotary = rotary

    def forward(self, positions, query, key):
        return RotaryEmbedding.forward_native(
            positions, query, key, self.rotary.head_dim,
            self.rotary.cos_sin_cache.to(query.dtype),
        )


class RoundedSquaredGate(nn.Module):
    """Keep HF's rounded squared-ReLU output before the existing product."""

    def __init__(self):
        super().__init__()
        self.activation = SquaredReLU()
        self.product = ProductGate()

    def forward(self, packed):
        gate, value = packed.chunk(2, dim=-1)
        return self.product(torch.cat((self.activation(gate), value), dim=-1))


class OfflineProjection(Linear):
    """Reuse quantization, then GEMM, then the loaded scalar for each projection.

    The native L4 folds scales into weights before GEMM, changing BF16 rounding.
    Its unchanged quantization helper is used in FP32, matching HF's opmath.
    This composition does not claim the native packed integer-kernel latency.
    """

    def __init__(self, width, out_sizes):
        super().__init__(width, sum(out_sizes), bias=False)
        self.out_sizes = out_sizes
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(1), requires_grad=False) for _ in out_sizes
        ])

    def forward(self, hidden):
        quantized = _fake_quant_act_bf16(hidden.float()).to(hidden.dtype)
        parts = super().forward(quantized).split(self.out_sizes, dim=-1)
        scaled = [part * scale for part, scale in zip(parts, self.scales)]
        return scaled[0] if len(scaled) == 1 else torch.cat(scaled, dim=-1)


def build_from_config(config, device, dtype):
    quant = config.quantization_config
    if (_tp_size() != 1 or config.hidden_act != "relu2" or config.attention_bias
            or not config.tie_word_embeddings or not config.use_cache
            or quant["linear_class"] != "autobitlinear"
            or quant["quantization_mode"] != "offline"
            or config.rope_parameters["rope_type"] != "default"):
        raise ValueError("This case preserves the documented offline-quantized BitNet checkpoint")
    adapted = BitNetConfig(**{key: getattr(config, key) for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "vocab_size", "max_position_embeddings",
        "rms_norm_eps", "tie_word_embeddings",
    )}, head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
        rope_theta=config.rope_parameters["rope_theta"], dtype=dtype)
    model = BitNetForCausalLM(adapted)
    rotary = NativeRotary(model.model.rotary_emb)
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
        layer.mlp.act_fn = RoundedSquaredGate()
        for parent, name in ((layer.self_attn, "qkv_proj"), (layer.self_attn, "o_proj"),
                             (layer.mlp, "gate_up_proj"), (layer.mlp, "down_proj")):
            old = getattr(parent, name)
            sizes = old.out_sizes if hasattr(old, "out_sizes") else [old.out_features]
            setattr(parent, name, OfflineProjection(old.in_features, sizes))
    # Pinned HF uses the split-half convention; the library also exposes it.
    model.model.rotary_emb.is_neox_style = True
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    if not torch.equal(remaining["model.embed_tokens.weight"], remaining["lm_head.weight"]):
        raise ValueError("BitNet's tied embedding and output weights disagree")
    model.model.embed_tokens.embedding_op.emb.weight.copy_(remaining.pop("model.embed_tokens.weight"))
    model.lm_head.embedding_op.emb.weight.copy_(remaining.pop("lm_head.weight"))
    model.model.norm.weight.copy_(remaining.pop("model.norm.weight"))
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm", "self_attn.attn_sub_norm", "mlp.ffn_sub_norm"):
            layer.get_submodule(name).weight.copy_(remaining.pop(prefix + name + ".weight"))
        for parent, module, sources in (
            ("self_attn", layer.self_attn.qkv_proj, (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"))),
            ("self_attn", layer.self_attn.o_proj, ((None, "o_proj"),)),
            ("mlp", layer.mlp.gate_up_proj, ((0, "gate_proj"), (1, "up_proj"))),
            ("mlp", layer.mlp.down_proj, ((None, "down_proj"),)),
        ):
            weights = []
            for shard, (_, source) in enumerate(sources):
                weight = remaining.pop(prefix + parent + "." + source + ".weight")
                # HF exports unpacked offline weights. Reject a dense random
                # matrix rather than silently quantizing a different model.
                if not bool(((weight == -1) | (weight == 0) | (weight == 1)).all()):
                    raise ValueError("BitNet requires common offline ternary weights and their scales")
                scale = remaining.pop(prefix + parent + "." + source + ".weight_scale")
                weights.append(weight)
                module.scales[shard].copy_(scale)
            module.weight.copy_(torch.cat(weights, dim=0))
    if remaining:
        raise KeyError(f"Unmapped BitNet state: {sorted(remaining)}")
