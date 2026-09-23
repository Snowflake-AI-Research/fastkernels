"""CTRL's sinusoidal positions and sequential, biased ReLU decoder."""

import math
import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.relu import ReLU
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp
from ..runner import Config
from .gpt_neox import DecoderBackbone, DecoderLM
from .phi import BiasedHeadProduct
from .llama import make_workloads as llama_workloads


def decoder_config(config):
    return Config(dict(config, hidden_size=config.n_embd, num_attention_heads=config.n_head,
                       num_hidden_layers=config.n_layer, intermediate_size=config.dff,
                       max_position_embeddings=config.n_positions, layer_norm_eps=config.layer_norm_epsilon))


class CTRLLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=1e-6, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=1e-6, promote_fp32=False)
        self.self_attn = LlamaAttention(config.hidden_size, config.num_attention_heads,
            config.num_attention_heads, config.hidden_size // config.num_attention_heads,
            bias=True, o_proj_bias=True, nope=True)
        self.mlp = VitEncoderMlp(config.hidden_size, config.intermediate_size, bias=True)
        self.mlp.act = ReLU()

    def forward(self, positions, hidden):
        hidden = hidden + self.self_attn(positions, self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class CTRLBackbone(DecoderBackbone):
    def __init__(self, config, layers):
        super().__init__(config, layers)
        self.embedding_scale = math.sqrt(config.hidden_size)
        # Positions are fixed metadata, built in FP32 as in the HF constructor.
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32, device="cpu")[:, None]
        channels = torch.arange(config.hidden_size, dtype=torch.float32, device="cpu")[None, :]
        angles = positions * (1 / torch.pow(10000, 2 * (channels // 2) / config.hidden_size))
        self.register_buffer("positions", torch.cat([angles[:, 0::2].sin(), angles[:, 1::2].cos()], -1), persistent=False)

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids) * self.embedding_scale + self.positions[positions]
        for layer in self.layers:
            hidden = layer(positions, hidden)
        return self.norm(hidden)


def build_from_config(config, device, dtype):
    if _tp_size() != 1 or not config.tie_word_embeddings or not config.use_cache:
        raise ValueError("CTRL's selected path uses tied embeddings and cached decoding")
    adapted = decoder_config(config)
    model = DecoderLM(adapted, [])
    model.model = CTRLBackbone(adapted, [CTRLLayer(adapted) for _ in range(adapted.num_hidden_layers)])
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    model.lm_head.linear_op = BiasedHeadProduct(config.n_embd, config.vocab_size, bias=True)
    model.lm_head.linear_op.weight = model.lm_head.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {"model.embed_tokens.embedding_op.emb.weight": remaining.pop("transformer.w.weight"),
              "lm_head.embedding_op.emb.weight": remaining.pop("lm_head.weight"),
              "lm_head.linear_op.bias": remaining.pop("lm_head.bias")}
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("CTRL's tied weights disagree")
    mapped["lm_head.linear_op.weight"] = mapped["lm_head.embedding_op.emb.weight"]
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("transformer.layernorm." + field)
    for index in range(config.n_layer):
        src, dst = f"transformer.h.{index}.", f"model.layers.{index}."
        for field in ("weight", "bias"):
            mapped[dst + "self_attn.qkv_proj." + field] = torch.cat([
                remaining.pop(src + f"multi_head_attention.W{part}." + field) for part in ("q", "k", "v")])
            for target, source in (("input_layernorm", "layernorm1"), ("post_attention_layernorm", "layernorm2"),
                                   ("self_attn.o_proj", "multi_head_attention.dense"), ("mlp.fc1", "ffn.0"), ("mlp.fc2", "ffn.2")):
                mapped[dst + target + "." + field] = remaining.pop(src + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped CTRL state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return llama_workloads(model, inputs, decoder_config(config), case=case)
