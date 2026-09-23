"""Hunyuan dense decoder: fixed-alpha RoPE followed by weighted per-head Q/K norm."""

from copy import copy

from fastkernels.hf_coverage.models.llama import load_state_dict_into as load_llama, make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.models.qwen2_precision import configure_language
from fastkernels.tasks.baseline.L1.rms_norm_native import RMSNormNative
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class ContiguousRotary(RotaryEmbedding):
    """Pack rotated Q/K slices for the existing post-RoPE norm's head view."""

    def forward(self, positions, query, key):
        query, key = self.forward_native(
            positions, query, key, self.head_dim, self.cos_sin_cache.to(query.dtype),
        )
        return query.contiguous(), key.contiguous()


def build_from_config(config, device, dtype):
    rope = config.rope_parameters
    if config.hidden_act != "silu" or config.attention_bias or not config.tie_word_embeddings:
        raise ValueError("Selected Hunyuan requires bias-free SiLU and a tied head")
    if rope["rope_type"] != "dynamic" or not rope.get("alpha"):
        raise ValueError("Selected Hunyuan uses its fixed-alpha dynamic RoPE initialization")
    fk = decoder_config(config, dtype)
    model = LlamaForCausalLM(fk)
    configure_language(model.model, config)
    base = rope["rope_theta"] * rope["alpha"] ** (fk.head_dim / (fk.head_dim - 2))
    rotary = ContiguousRotary(fk.head_dim, config.max_position_embeddings, base)
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
        # Existing post-RoPE normalization hooks accept weighted RMSNorm too.
        layer.self_attn.q_wl_norm = RMSNormNative(fk.head_dim, config.rms_norm_eps)
        layer.self_attn.k_wl_norm = RMSNormNative(fk.head_dim, config.rms_norm_eps)
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


def load_state_dict_into(model, state_dict, config):
    names = {f"model.layers.{i}.self_attn.{name}.weight"
             for i in range(config.num_hidden_layers) for name in ("query_layernorm", "key_layernorm")}
    if not names <= state_dict.keys():
        raise KeyError("Hunyuan Q/K normalization weights missing")
    carrier = copy(config)
    carrier.head_dim = model.config.head_dim
    load_llama(model, {k: v for k, v in state_dict.items() if k not in names}, carrier)
    for i, layer in enumerate(model.model.layers):
        for name, target in (("query_layernorm", "q_wl_norm"), ("key_layernorm", "k_wl_norm")):
            parameter = getattr(layer.self_attn, target).weight
            source = state_dict[f"model.layers.{i}.self_attn.{name}.weight"]
            if source.shape != parameter.shape:
                raise ValueError("Hunyuan normalization width mismatch")
            parameter.data.copy_(source)
