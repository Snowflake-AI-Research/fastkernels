"""Phi's shared-normalization parallel block and biased language-model head."""

import torch
from torch import nn

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.t5_dense import NewGELUActivation
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .gpt_neox import DecoderLM
from .llama import make_workloads as decoder_workloads
from .stablelm import NativePartialRotary
from .qwen2_precision import DenseCachedAttention


class PhiLayer(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        width, heads = config.hidden_size, config.num_attention_heads
        self.input_layernorm = LayerNorm(width, eps=config.layer_norm_eps, promote_fp32=False)
        self.self_attn = LlamaAttention(width, heads, config.num_key_value_heads, width // heads,
                                        rotary_emb=rotary, bias=True, o_proj_bias=True)
        self.self_attn.attn = DenseCachedAttention(heads, config.num_key_value_heads, width // heads)
        self.mlp = VitEncoderMlp(width, config.intermediate_size)
        self.mlp.act = NewGELUActivation()

    def forward(self, positions, hidden):
        normalized = self.input_layernorm(hidden)
        return self.self_attn(positions, normalized) + self.mlp(normalized) + hidden


class BiasedHeadProduct(Linear):
    """Use the existing biased linear operation through the workload's head interface."""

    def forward(self, hidden, embedding_weight):
        # The head weight is owned here; the generic workload argument is its alias.
        return super().forward(hidden)


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if (_tp_size() != 1 or config.hidden_act != "gelu_new" or config.qk_layernorm
            or config.tie_word_embeddings or not config.use_cache
            or rope["rope_type"] != "default" or rope["partial_rotary_factor"] != 0.5):
        raise ValueError("The selected Phi checkpoint uses shared affine norm, gelu_new and half-head RoPE without QK norm")
    head_dim = config.hidden_size // config.num_attention_heads
    with torch.device("cpu"):
        rotary = NativePartialRotary(head_dim, head_dim // 2, config.max_position_embeddings, rope["rope_theta"])
    model = DecoderLM(config, [PhiLayer(config, rotary) for _ in range(config.num_hidden_layers)])
    model.lm_head.linear_op = BiasedHeadProduct(config.hidden_size, config.vocab_size)
    model.lm_head.linear_op.weight = model.lm_head.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {
        "model.embed_tokens.embedding_op.emb.weight": remaining.pop("model.embed_tokens.weight"),
        "lm_head.embedding_op.emb.weight": remaining["lm_head.weight"],
        "lm_head.linear_op.weight": remaining.pop("lm_head.weight"),
        "lm_head.linear_op.bias": remaining.pop("lm_head.bias"),
    }
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("model.final_layernorm." + field)
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        for field in ("weight", "bias"):
            mapped[prefix + "self_attn.qkv_proj." + field] = torch.cat([
                remaining.pop(prefix + f"self_attn.{part}_proj." + field) for part in ("q", "k", "v")])
            for target, source in (("input_layernorm", "input_layernorm"),
                                   ("self_attn.o_proj", "self_attn.dense"),
                                   ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
                mapped[prefix + target + "." + field] = remaining.pop(prefix + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped Phi state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return decoder_workloads(model, inputs, config, case=case)
