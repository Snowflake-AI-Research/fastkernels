"""GPT-Neo's learned positions and alternating global/local unscaled attention."""

import torch

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L4.llama import LlamaConfig

from .gpt2 import LearnedPositionCausalLM
from .gptj import Float32ScoreAttention
from .llama import make_workloads as decoder_workloads


def build_from_config(config, device, dtype):
    if (_tp_size() != 1 or config.activation_function != "gelu_new"
            or not config.tie_word_embeddings or not config.use_cache):
        raise ValueError("The GPT-Neo example uses cached tied decoding with gelu_new")
    pattern = config.attention_layers
    if len(pattern) != config.num_layers or set(pattern) != {"local", "global"}:
        raise ValueError("Preserve GPT-Neo's global and local attention layers")
    adapted = LlamaConfig(hidden_size=config.hidden_size,
                          intermediate_size=config.intermediate_size or 4 * config.hidden_size,
                          num_hidden_layers=config.num_layers, num_attention_heads=config.num_heads,
                          num_key_value_heads=config.num_heads, head_dim=config.hidden_size // config.num_heads,
                          vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
                          rms_norm_eps=config.layer_norm_epsilon, dtype=dtype)
    model = LearnedPositionCausalLM(adapted, "gelu_new")
    for layer, kind in zip(model.model.layers, pattern):
        layer.self_attn = LlamaAttention(config.hidden_size, config.num_heads, config.num_heads,
                                         adapted.head_dim, nope=True, o_proj_bias=True,
                                         sliding_window=config.window_size if kind == "local" else None)
        # HF keeps QK and softmax in FP32, then stores probabilities in BF16.
        # Reuse the same explicit score composition as CodeGen, without scaling.
        layer.self_attn.attn = Float32ScoreAttention(config.num_heads, adapted.head_dim)
        layer.self_attn.attn.scale = 1.0
        layer.self_attn.attn.sliding_window = config.window_size if kind == "local" else None
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    remaining = dict(state_dict)
    mapped = {
        "model.embed_tokens.embedding_op.emb.weight": remaining.pop("transformer.wte.weight"),
        "model.position_embedding.emb.weight": remaining.pop("transformer.wpe.weight"),
        "lm_head.embedding_op.emb.weight": remaining.pop("lm_head.weight"),
    }
    if not torch.equal(mapped["model.embed_tokens.embedding_op.emb.weight"], mapped["lm_head.embedding_op.emb.weight"]):
        raise ValueError("GPT-Neo's tied embedding and head disagree")
    for field in ("weight", "bias"):
        mapped["model.norm." + field] = remaining.pop("transformer.ln_f." + field)
    for index in range(config.num_layers):
        src, dst = f"transformer.h.{index}.", f"model.layers.{index}."
        mapped[dst + "self_attn.qkv_proj.weight"] = torch.cat([
            remaining.pop(src + f"attn.attention.{part}_proj.weight") for part in ("q", "k", "v")])
        for field in ("weight", "bias"):
            for target, source in (("input_layernorm", "ln_1"), ("post_attention_layernorm", "ln_2"),
                                   ("self_attn.o_proj", "attn.attention.out_proj"),
                                   ("mlp.fc1", "mlp.c_fc"), ("mlp.fc2", "mlp.c_proj")):
                mapped[dst + target + "." + field] = remaining.pop(src + source + "." + field)
    if remaining:
        raise KeyError(f"Unmapped GPT-Neo state: {sorted(remaining)}")
    model.load_state_dict(mapped, strict=True)


def make_workloads(model, inputs, config, *, case=None):
    return decoder_workloads(model, inputs, model.config, case=case)
