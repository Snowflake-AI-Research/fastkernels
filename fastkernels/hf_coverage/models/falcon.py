"""Falcon-7B's single-query-group attention and shared-normalization parallel block."""

from fastkernels.hf_coverage.runner import config_values
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.vit_encoder_mlp import VitEncoderMlp

from .gpt_neox import DecoderLM
from .gptj import GPTJLayer
from .llama import make_workloads as decoder_workloads


def build_from_config(config, device, dtype):
    if (_tp_size() != 1 or config.alibi or config.bias or config.new_decoder_architecture
            or not config.multi_query or not config.parallel_attn or not config.use_cache):
        raise ValueError("The selected Falcon-7B checkpoint uses bias-free MQA, shared norm and parallel residuals with RoPE")
    adapted = config_values(dict(config))
    adapted.update(intermediate_size=4 * config.hidden_size, layer_norm_eps=config.layer_norm_epsilon)
    head_dim = config.hidden_size // config.num_attention_heads
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The selected Falcon checkpoint uses default full-head RoPE")
    rotary = RotaryEmbedding(head_dim, config.max_position_embeddings, rope["rope_theta"])
    layers = []
    for _ in range(config.num_hidden_layers):
        layer = GPTJLayer(adapted, rotary)
        layer.self_attn = LlamaAttention(config.hidden_size, config.num_attention_heads, 1, head_dim, rotary_emb=rotary)
        layer.mlp = VitEncoderMlp(config.hidden_size, 4 * config.hidden_size, bias=False)
        layers.append(layer)
    return DecoderLM(adapted, layers).to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {
        "model.embed_tokens.embedding_op.emb.weight": remaining.pop("transformer.word_embeddings.weight"),
        "lm_head.embedding_op.emb.weight": remaining.pop("lm_head.weight"),
    }
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("transformer.ln_f." + field)
    for index in range(config.num_hidden_layers):
        src, dst = f"transformer.h.{index}.", f"model.layers.{index}."
        for field in ("weight", "bias"):
            mapped[dst + "input_layernorm." + field] = remaining.pop(src + "input_layernorm." + field)
        for target, source in (("self_attn.qkv_proj", "self_attention.query_key_value"),
                               ("self_attn.o_proj", "self_attention.dense"),
                               ("mlp.fc1", "mlp.dense_h_to_4h"), ("mlp.fc2", "mlp.dense_4h_to_h")):
            mapped[dst + target + ".weight"] = remaining.pop(src + source + ".weight")
    if remaining:
        raise KeyError(f"Unmapped Falcon state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return decoder_workloads(model, inputs, model.config, case=case)
