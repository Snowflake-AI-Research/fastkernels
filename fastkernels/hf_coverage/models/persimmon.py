"""Persimmon's per-head Q/K LayerNorm and squared-ReLU sequential decoder."""

from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .gpt_neox import DecoderLM, unpack_head_qkv
from .llama import make_workloads
from .nemotron import NemotronLayer
from .stablelm import PartialRotary


class PersimmonLayer(NemotronLayer):
    def __init__(self, config, rotary):
        nn.Module.__init__(self)
        width, heads = config.hidden_size, config.num_attention_heads
        head_dim = width // heads
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.self_attn = LlamaAttention(width, heads, heads, head_dim, rotary_emb=rotary, bias=True, o_proj_bias=True)
        self.self_attn.q_norm = LayerNorm(head_dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.self_attn.k_norm = LayerNorm(head_dim, eps=config.layer_norm_eps, promote_fp32=False)
        self.mlp = VitEncoderMlp(width, config.intermediate_size)
        self.mlp.act = SquaredReLU()


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or config.hidden_act != "relu2" or not config.qk_layernorm
            or config.tie_word_embeddings or not config.use_cache
            or rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 0.5):
        raise ValueError("Persimmon's selected checkpoint uses Q/K LayerNorm, squared ReLU and half-head RoPE")
    head_dim = config.hidden_size // config.num_attention_heads
    rotary = PartialRotary(head_dim, head_dim // 2, config.max_position_embeddings, rope["rope_theta"])
    return DecoderLM(config, [PersimmonLayer(config, rotary) for _ in range(config.num_hidden_layers)]).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {"model.embed_tokens.embedding_op.emb.weight": remaining.pop("model.embed_tokens.weight"),
              "lm_head.embedding_op.emb.weight": remaining.pop("lm_head.weight")}
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("model.final_layernorm." + field)
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        for field in ("weight", "bias"):
            mapped[prefix + "self_attn.qkv_proj." + field] = unpack_head_qkv(
                remaining.pop(prefix + "self_attn.query_key_value." + field), config.num_attention_heads)
            for target, source in (("input_layernorm", "input_layernorm"),
                                   ("post_attention_layernorm", "post_attention_layernorm"),
                                   ("self_attn.q_norm", "self_attn.q_layernorm"),
                                   ("self_attn.k_norm", "self_attn.k_layernorm"),
                                   ("self_attn.o_proj", "self_attn.dense"),
                                   ("mlp.fc1", "mlp.dense_h_to_4h"), ("mlp.fc2", "mlp.dense_4h_to_h")):
                mapped[prefix + target + "." + field] = remaining.pop(prefix + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped Persimmon state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
