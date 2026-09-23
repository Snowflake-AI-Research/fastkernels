"""NanoChat's weightless normalization, reversed RoPE and softcapped logits."""

from contextlib import nullcontext

import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L1.tanh import Tanh
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaModel

from .llama import make_workloads
from .qwen2_precision import DenseCachedAttention


class WeightlessNorm(RMSNorm):
    """Reuse native RMSNorm with the model's explicit BF16 residual boundary."""

    def __init__(self, width, eps):
        super().__init__(width, eps=eps, elementwise_affine=False)

    def forward(self, hidden, residual=None):
        if residual is None:
            return self.forward_native(hidden, None, self.eps, self.hidden_size)
        residual = hidden + residual
        return self.forward_native(residual, None, self.eps, self.hidden_size), residual


def position_table(config, device, dtype):
    """Prepare fixed positions using HF's CPU frequency initialization."""
    dim = config.hidden_size // config.num_attention_heads
    exponent = torch.arange(0, dim, 2, device="cpu", dtype=torch.float32) / dim
    frequencies = (1.0 / config.rope_parameters["rope_theta"] ** exponent).to(device)
    positions = torch.arange(config.max_position_embeddings, device=device, dtype=torch.float32)
    device_type = torch.device(device).type
    context = nullcontext() if device_type == "meta" else torch.autocast(device_type=device_type, enabled=False)
    with context:
        phases = (frequencies[None, :, None] @ positions[None, None, :]).transpose(1, 2)
        phases = torch.cat((phases, phases), dim=-1)
        cosine, sine = phases.cos().to(dtype), phases.sin().to(dtype)
    # NanoChat rotates in the opposite direction to the existing RoPE operator.
    return torch.cat((cosine[0, :, :dim // 2], -sine[0, :, :dim // 2]), dim=-1)


class NanoChatBackbone(LlamaModel):
    def forward(self, input_ids, positions):
        embeddings = self.norm(self.embed_tokens(input_ids))
        return super().forward(input_ids, positions, inputs_embeds=embeddings)


class SoftcappedProjection(nn.Module):
    def __init__(self, product, cap):
        super().__init__()
        self.product, self.cap, self.tanh = product, cap, Tanh()

    def forward(self, hidden, weight):
        return self.tanh(self.product(hidden, weight) / self.cap) * self.cap


class ContiguousRotary(nn.Module):
    """Retain HF's rounded products using the existing native rotary callable."""

    def __init__(self, rotary):
        super().__init__()
        self.rotary = rotary

    def forward(self, positions, query, key):
        query, key = self.rotary.forward_native(
            positions, query, key, self.rotary.head_dim,
            self.rotary.cos_sin_cache.to(query.dtype),
        )
        return query.contiguous(), key.contiguous()


class NanoChatLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = NanoChatBackbone(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or config.hidden_act != "relu2" or config.attention_bias
            or config.tie_word_embeddings or not config.use_cache or rope["rope_type"] != "default"
            or config.final_logit_softcapping is None):
        raise ValueError("NanoChat's default path uses bias-free squared ReLU, weightless RMSNorm and capped untied logits")
    head_dim = config.hidden_size // config.num_attention_heads
    adapted = LlamaConfig(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                          num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
                          num_key_value_heads=config.num_key_value_heads, head_dim=head_dim,
                          vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
                          rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
                          rope_scaling_factor=1.0, dtype=dtype)
    model = NanoChatLM(adapted)
    # CPU versus GPU frequency initialization changes a few BF16 table entries.
    # Match the ordinary HF loader before reusing the existing rotary operation.
    model.model.rotary_emb.cos_sin_cache = position_table(config, device, dtype)
    model.model.rotary_emb = ContiguousRotary(model.model.rotary_emb)
    model.model.norm = WeightlessNorm(config.hidden_size, config.rms_norm_eps)
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = model.model.rotary_emb
        layer.self_attn.attn = DenseCachedAttention(
            config.num_attention_heads, config.num_key_value_heads, head_dim,
        )
        layer.input_layernorm = WeightlessNorm(config.hidden_size, config.rms_norm_eps)
        layer.post_attention_layernorm = WeightlessNorm(config.hidden_size, config.rms_norm_eps)
        layer.self_attn.q_wl_norm = WeightlessNorm(head_dim, config.rms_norm_eps)
        layer.self_attn.k_wl_norm = WeightlessNorm(head_dim, config.rms_norm_eps)
        layer.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size, bias=False)
        layer.mlp.act = SquaredReLU()
    model.lm_head.linear_op = SoftcappedProjection(model.lm_head.linear_op, config.final_logit_softcapping)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    mapped = dict(state_dict)
    mapped["model.embed_tokens.embedding_op.emb.weight"] = mapped.pop("model.embed_tokens.weight")
    mapped["lm_head.embedding_op.emb.weight"] = mapped.pop("lm_head.weight")
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}.self_attn."
        mapped[prefix + "qkv_proj.weight"] = torch.cat([
            mapped.pop(prefix + part + "_proj.weight") for part in ("q", "k", "v")])
    model.load_state_dict(mapped, strict=True)
