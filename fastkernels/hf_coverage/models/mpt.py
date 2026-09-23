"""MPT constructor-default ALiBi decoder with whole-sequence inference."""

import math
import torch
from torch import nn

from fastkernels.hf_coverage.runner import config_values
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .bloom import AlibiAttention
from .gpt_neox import DecoderLM
from .llama import make_workloads as decoder_workloads


class MptAlibiAttention(AlibiAttention):
    def __init__(self, heads, head_dim, bias_max, max_length):
        super().__init__(heads, head_dim)
        self.bias_max, self.max_length = bias_max, max_length

    def _sdpa_one(self, query, key, value, key_offset=0):
        # Position-only metadata follows MPT's distinct slope order and origin.
        power = 2 ** math.ceil(math.log2(self.num_heads))
        exponents = torch.arange(1, power + 1, device=query.device).float() * (self.bias_max / power)
        slopes = 1.0 / torch.pow(2, exponents)
        if power != self.num_heads:
            slopes = torch.cat((slopes[1::2], slopes[::2]))[:self.num_heads]
        key_positions = torch.arange(key.shape[0], device=query.device)
        query_positions = key_offset + torch.arange(query.shape[0], device=query.device)
        bias = slopes[:, None, None] * (key_positions - key.shape[0] + 1)
        mask = torch.zeros(query.shape[0], key.shape[0], device=query.device, dtype=query.dtype)
        mask = mask.masked_fill(key_positions[None, :] > query_positions[:, None], torch.finfo(query.dtype).min)
        scores = self.qk(query.transpose(0, 1), key.permute(1, 2, 0)) * self.scale
        probabilities = self.softmax(scores.float() + bias + mask).to(value.dtype)
        return self.pv(probabilities, value.transpose(0, 1)).transpose(0, 1)


class MptLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, heads = config.hidden_size, config.num_attention_heads
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, create_offset=False, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(width, eps=config.layer_norm_eps, create_offset=False, promote_fp32=False)
        self.self_attn = LlamaAttention(width, heads, heads, width // heads, nope=True)
        self.self_attn.attn = MptAlibiAttention(heads, width // heads, config.attn_config.alibi_bias_max, config.max_seq_len)
        self.mlp = VitEncoderMlp(width, 4 * width, bias=False)

    def forward(self, positions, hidden):
        hidden = hidden + self.self_attn(positions, self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


def build_from_config(config, device, dtype):
    attention = config.attn_config
    if (_tp_size() != 1 or config.use_cache or not config.tie_word_embeddings or not config.no_bias
            or not attention.alibi or attention.clip_qkv is not None or attention.qk_ln
            or attention.attn_type != "multihead_attention" or attention.prefix_lm
            or attention.attn_uses_sequence_id or attention.softmax_scale is not None):
        raise ValueError("This MPT case preserves constructor-default noncached multihead ALiBi inference")
    adapted = config_values(dict(config))
    adapted.update(hidden_size=config.d_model, num_attention_heads=config.n_heads,
                   num_hidden_layers=config.n_layers, layer_norm_eps=config.layer_norm_epsilon,
                   max_position_embeddings=config.max_seq_len)
    model = DecoderLM(adapted, [MptLayer(adapted) for _ in range(config.n_layers)])
    model.model.norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon, create_offset=False, promote_fp32=False)
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {
        "model.embed_tokens.embedding_op.emb.weight": remaining.pop("transformer.wte.weight"),
        "lm_head.embedding_op.emb.weight": remaining.pop("lm_head.weight"),
        "model.norm.weight": remaining.pop("transformer.norm_f.weight"),
    }
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("MPT's tied embedding and head disagree")
    for index in range(config.n_layers):
        src, dst = f"transformer.blocks.{index}.", f"model.layers.{index}."
        for target, source in (("input_layernorm", "norm_1"), ("post_attention_layernorm", "norm_2"),
                               ("self_attn.qkv_proj", "attn.Wqkv"), ("self_attn.o_proj", "attn.out_proj"),
                               ("mlp.fc1", "ffn.up_proj"), ("mlp.fc2", "ffn.down_proj")):
            mapped[dst + target + ".weight"] = remaining.pop(src + source + ".weight")
    if remaining:
        raise KeyError(f"Unmapped MPT state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config):
    return decoder_workloads(model, inputs, model.config, cached_decode=False)
