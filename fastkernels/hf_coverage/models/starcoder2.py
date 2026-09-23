"""StarCoder2 using biased attention, affine LayerNorm, and existing GELU MLP."""

import torch

from fastkernels.hf_coverage.models.llama import make_workloads as llama_workloads
from fastkernels.hf_coverage.models.olmo import ResidualLayerNorm
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.tasks.baseline.L1.gelu import GELU
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.parallel_linear import RowParallelLinear
from fastkernels.tasks.baseline.L2.vision_mlp import VisionMLP
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


def build_from_config(config, device, dtype):
    if config.hidden_act != "gelu_pytorch_tanh" or not config.use_bias:
        raise ValueError("Selected StarCoder2 requires biased projections and tanh-GELU")
    if config.rope_parameters["rope_type"] != "default":
        raise ValueError("Selected StarCoder2 requires default RoPE")
    fk = decoder_config(config, dtype)
    fk.qkv_bias = config.use_bias
    model = LlamaForCausalLM(fk)
    for layer in model.model.layers:
        layer.input_layernorm = ResidualLayerNorm(config.hidden_size, config.norm_epsilon, promote_fp32=False)
        layer.post_attention_layernorm = ResidualLayerNorm(config.hidden_size, config.norm_epsilon, promote_fp32=False)
        layer.mlp = VisionMLP(config.hidden_size, config.intermediate_size, act_fn=GELU(approximate="tanh"), bias=True)
        layer.self_attn.o_proj = RowParallelLinear(fk.num_attention_heads * fk.head_dim, config.hidden_size, bias=True)
        a = layer.self_attn.attn
        layer.self_attn.attn = Attention(a.num_heads, a.head_size, a.scale, num_kv_heads=a.num_kv_heads,
                                      sliding_window=config.sliding_window)
    model.model.norm = ResidualLayerNorm(config.hidden_size, config.norm_epsilon, promote_fp32=False)
    if config.tie_word_embeddings:
        model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight, "model.norm.bias": model.model.norm.bias,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            for suffix in ("weight", "bias"):
                direct[p + name + "." + suffix] = getattr(getattr(layer, name), suffix)
        for suffix in ("weight", "bias"):
            direct[p + "self_attn.o_proj." + suffix] = getattr(layer.self_attn.o_proj, suffix)
            for name, target in (("c_fc", "fc1"), ("c_proj", "fc2")):
                direct[p + "mlp." + name + "." + suffix] = getattr(getattr(layer.mlp, target), suffix)
            for shard in ("q", "k", "v"):
                packed[p + f"self_attn.{shard}_proj." + suffix] = (getattr(layer.self_attn.qkv_proj, suffix), shard)
    if set(state_dict) != direct.keys() | packed.keys():
        raise KeyError("StarCoder2 state keys do not match the complete decoder")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"StarCoder2 weight shape mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)


def make_workloads(model, inputs, config, *, case=None):
    return llama_workloads(model, inputs, config, case=case,
                           cache_windows=[config.sliding_window] * config.num_hidden_layers)
