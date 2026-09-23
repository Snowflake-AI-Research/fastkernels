"""Dense ERNIE 4.5 causal LM with a tied head and FP32 interleaved RoPE."""

import torch

from fastkernels.hf_coverage.models.llama import load_state_dict_into as load_llama_weights
from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import NativeFP32Rotary
from fastkernels.hf_coverage.models.qwen2_precision import configure_language
from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L4.llama import LlamaConfig, LlamaForCausalLM


class NativeInterleavedFP32Rotary(NativeFP32Rotary):
    """Reuse interleaved rotation with FP32 products and fixed HF positions."""

    def forward(self, positions, query, key):
        q, k = self.forward_native_interleaved(
            positions, query.float(), key.float(), self.head_dim, self.cos_sin_cache,
        )
        return q.to(query.dtype), k.to(key.dtype)


def build_from_config(config, device, dtype):
    if _tp_size() != 1:
        raise ValueError("The ERNIE 4.5 coverage workload requires tensor parallel size 1")
    if config.hidden_act != "silu" or config.use_bias or not config.tie_word_embeddings:
        raise ValueError("The ERNIE 4.5 checkpoint requires bias-free SiLU layers and a tied head")
    if config.num_attention_heads != 8 * config.num_key_value_heads:
        raise ValueError("The ERNIE 4.5 checkpoint preserves 8:1 grouped-query attention")
    if config.num_attention_heads * config.head_dim != 2 * config.hidden_size:
        raise ValueError("The ERNIE 4.5 checkpoint preserves attention width twice the hidden size")
    rope = config.rope_parameters
    if rope["rope_type"] != "default":
        raise ValueError("The selected ERNIE 4.5 checkpoint uses default RoPE")
    fk_config = LlamaConfig(
        hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim,
        vocab_size=config.vocab_size, max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=rope["rope_theta"],
        rope_scaling_factor=1.0, rope_low_freq_factor=1.0, rope_high_freq_factor=1.0,
        rope_original_max_position_embeddings=config.max_position_embeddings,
        dtype=dtype, qkv_bias=False,
    )
    model = LlamaForCausalLM(fk_config)
    configure_language(model.model, config)
    model = model.to(device=device, dtype=dtype).eval()
    rotary = NativeInterleavedFP32Rotary(
        config.head_dim, config.max_position_embeddings, rope["rope_theta"], device,
    ).to(device=device)
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("The ERNIE 4.5 tied embedding and output-head weights must agree")
    load_llama_weights(model, state_dict, config)
