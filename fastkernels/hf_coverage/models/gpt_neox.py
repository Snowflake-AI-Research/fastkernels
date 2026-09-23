"""GPT-NeoX parallel residual decoder, assembled from existing operations."""

import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from ..patches.gelu_fast import FastGELU
from .llama import make_workloads
from .stablelm import PartialRotary


class NeoXLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        width, heads = config.hidden_size, config.num_attention_heads
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.self_attn = LlamaAttention(width, heads, heads, width // heads,
                                        rotary_emb=rotary, bias=True, o_proj_bias=True)
        self.mlp = VitEncoderMlp(width, config.intermediate_size)
        self.mlp.act = FastGELU()

    def forward(self, positions, hidden):
        attention = self.self_attn(positions, self.input_layernorm(hidden))
        mlp = self.mlp(self.post_attention_layernorm(hidden))
        return mlp + attention + hidden


class DecoderBackbone(nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, promote_fp32=False)

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(positions, hidden)
        return self.norm(hidden)


class DecoderLM(nn.Module):
    def __init__(self, config, layers):
        super().__init__()
        self.config = config
        self.model = DecoderBackbone(config, layers)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.embedding_op.emb.weight = self.model.embed_tokens.embedding_op.emb.weight


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or not config.use_parallel_residual or not config.attention_bias
            or config.hidden_act != "gelu_fast" or config.tie_word_embeddings or not config.use_cache
            or rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 0.25):
        raise ValueError("The documented NeoX checkpoint uses parallel residuals, biased FastGELU blocks and quarter-head RoPE")
    head_dim = config.hidden_size // config.num_attention_heads
    rotary = PartialRotary(head_dim, head_dim // 4, config.max_position_embeddings, rope["rope_theta"])
    model = DecoderLM(config, [NeoXLayer(config, rotary) for _ in range(config.num_hidden_layers)])
    return model.to(device=device, dtype=dtype).eval()


def unpack_head_qkv(value, heads):
    """Convert HF's per-head Q/K/V rows into the existing projection's Q|K|V rows."""
    return value.view(heads, 3, -1, *value.shape[1:]).transpose(0, 1).reshape(value.shape)


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {
        "model.embed_tokens.embedding_op.emb.weight": remaining.pop("gpt_neox.embed_in.weight"),
        "lm_head.embedding_op.emb.weight": remaining.pop("embed_out.weight"),
    }
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("gpt_neox.final_layer_norm." + field)
    for index in range(config.num_hidden_layers):
        src, dst = f"gpt_neox.layers.{index}.", f"model.layers.{index}."
        for field in ("weight", "bias"):
            mapped[dst + "self_attn.qkv_proj." + field] = unpack_head_qkv(
                remaining.pop(src + "attention.query_key_value." + field), config.num_attention_heads)
            for target, source in (("input_layernorm", "input_layernorm"),
                                   ("post_attention_layernorm", "post_attention_layernorm"),
                                   ("self_attn.o_proj", "attention.dense"),
                                   ("mlp.fc1", "mlp.dense_h_to_4h"),
                                   ("mlp.fc2", "mlp.dense_4h_to_h")):
                mapped[dst + target + "." + field] = remaining.pop(src + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped NeoX state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
