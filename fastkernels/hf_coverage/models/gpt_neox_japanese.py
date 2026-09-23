"""Japanese NeoX's sequential residual blocks and final attention bias."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import config_values
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .gpt_neox import DecoderLM, unpack_head_qkv
from .llama import make_workloads


class JapaneseLayer(nn.Module):
    def __init__(self, config, rotary, last):
        super().__init__()
        width, heads = config.hidden_size, config.num_attention_heads
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.self_attn = LlamaAttention(width, heads, heads, width // heads, rotary_emb=rotary)
        self.attention_bias = nn.Parameter(torch.zeros(width)) if last else None
        self.mlp = VitEncoderMlp(width, width * config.intermediate_multiple_size, bias=False)

    def forward(self, positions, hidden):
        attention = self.self_attn(positions, self.input_layernorm(hidden))
        if self.attention_bias is not None:
            attention = attention + self.attention_bias
        hidden = hidden + attention
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or config.hidden_act != "gelu" or not config.tie_word_embeddings
            or not config.use_cache or rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 1.0):
        raise ValueError("The Japanese NeoX checkpoint uses exact GELU, tied embeddings and full-head RoPE")
    rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads,
                              config.max_position_embeddings, rope["rope_theta"])
    layers = [JapaneseLayer(config, rotary, i == config.num_hidden_layers - 1) for i in range(config.num_hidden_layers)]
    return DecoderLM(config, layers).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {
        "model.embed_tokens.embedding_op.emb.weight": remaining.pop("gpt_neox_japanese.embed_in.weight"),
        "lm_head.embedding_op.emb.weight": remaining.pop("embed_out.weight"),
    }
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("Japanese NeoX's tied embedding weights disagree")
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("gpt_neox_japanese.final_layer_norm." + field)
    for index in range(config.num_hidden_layers):
        src, dst = f"gpt_neox_japanese.layers.{index}.", f"model.layers.{index}."
        mapped[dst + "self_attn.qkv_proj.weight"] = unpack_head_qkv(
            remaining.pop(src + "attention.query_key_value.weight"), config.num_attention_heads)
        for name in ("input_layernorm", "post_attention_layernorm"):
            for field in ("weight", "bias"):
                mapped[dst + name + "." + field] = remaining.pop(src + name + "." + field)
        for target, source in (("self_attn.o_proj", "attention.dense"),
                               ("mlp.fc1", "mlp.dense_h_to_4h"), ("mlp.fc2", "mlp.dense_4h_to_h")):
            mapped[dst + target + ".weight"] = remaining.pop(src + source + ".weight")
        if index == config.num_hidden_layers - 1:
            mapped[dst + "attention_bias"] = remaining.pop(src + "attention.dense_bias")
    if remaining:
        raise KeyError(f"Unmapped Japanese NeoX state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
