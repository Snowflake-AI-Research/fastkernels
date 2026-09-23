"""Nemotron's squared-ReLU blocks and fixed one-plus LayerNorm gains."""

import torch
from torch import nn

from fastkernels.hf_coverage.runner import config_values
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.squared_relu import SquaredReLU
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .gpt_neox import DecoderLM
from .llama import make_workloads
from .stablelm import PartialRotary


class NemotronLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        width = config.hidden_size
        self.input_layernorm = LayerNorm(width, eps=config.norm_eps, promote_fp32=False)
        self.post_attention_layernorm = LayerNorm(width, eps=config.norm_eps, promote_fp32=False)
        self.self_attn = LlamaAttention(width, config.num_attention_heads, config.num_key_value_heads,
                                        config.head_dim, rotary_emb=rotary)
        self.mlp = VitEncoderMlp(width, config.intermediate_size, bias=False)
        self.mlp.act = SquaredReLU()

    def forward(self, positions, hidden):
        hidden = hidden + self.self_attn(positions, self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or config.hidden_act != "relu2" or config.tie_word_embeddings
            or not config.use_cache or rope["rope_type"] != "default"
            or rope["partial_rotary_factor"] != 0.5):
        raise ValueError("The selected Minitron checkpoint uses squared ReLU, untied head and half-head RoPE")
    adapted = config_values(dict(config))
    adapted.layer_norm_eps = config.norm_eps
    rotary = PartialRotary(config.head_dim, config.head_dim // 2,
                           config.max_position_embeddings, rope["rope_theta"])
    return DecoderLM(adapted, [NemotronLayer(config, rotary) for _ in range(config.num_hidden_layers)]).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {"model.embed_tokens.embedding_op.emb.weight": remaining.pop("model.embed_tokens.weight"),
              "lm_head.embedding_op.emb.weight": remaining.pop("lm_head.weight")}
    # This inference-constant parameter transform preserves HF's native-dtype addition.
    for field in ("weight", "bias"):
        value = remaining.pop("model.norm." + field)
        mapped["model.norm." + field] = value + 1 if field == "weight" else value
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            for field in ("weight", "bias"):
                value = remaining.pop(prefix + norm + "." + field)
                mapped[prefix + norm + "." + field] = value + 1 if field == "weight" else value
        mapped[prefix + "self_attn.qkv_proj.weight"] = torch.cat([
            remaining.pop(prefix + f"self_attn.{part}_proj.weight") for part in ("q", "k", "v")])
        for target, source in (("self_attn.o_proj", "self_attn.o_proj"),
                               ("mlp.fc1", "mlp.up_proj"), ("mlp.fc2", "mlp.down_proj")):
            mapped[prefix + target + ".weight"] = remaining.pop(prefix + source + ".weight")
    if remaining:
        raise KeyError(f"Unmapped Nemotron state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)
