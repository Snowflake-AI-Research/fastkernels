"""MiniMax M2 with native FP8 projections/experts and joint Q/K normalization."""

from types import SimpleNamespace

import torch

from fastkernels.hf_coverage.models.deepseek_v3 import NativeFP8Experts, load_mapped_state, prepare_linears
from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import JointNorm, decoder_config
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


def build_from_config(config, device, dtype):
    if config.quantization_config["quant_method"] != "fp8" or config.output_router_logits:
        raise ValueError("Selected MiniMax M2 uses FP8 and does not return router outputs")
    model = LlamaForCausalLM(decoder_config(config, dtype))
    expert_config = SimpleNamespace(
        hidden_size=config.hidden_size, n_routed_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok, moe_intermediate_size=config.intermediate_size,
        n_shared_experts=0, n_group=1, topk_group=1, scoring_func="sigmoid", topk_method="noaux_tc",
        norm_topk_prob=True, routed_scaling_factor=1.0,
    )
    for layer in model.model.layers:
        layer.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim,
            rotary_emb=model.model.rotary_emb, quant_config=config.quantization_config,
        )
        layer.self_attn.q_norm = JointNorm(config.num_attention_heads * config.head_dim, config.rms_norm_eps)
        layer.self_attn.k_norm = JointNorm(config.num_key_value_heads * config.head_dim, config.rms_norm_eps)
        layer.mlp = NativeFP8Experts(expert_config, config.quantization_config, epsilon=0.0, gate_fp32=False)
    return model.to(device=device).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            direct[p + name + ".weight"] = getattr(layer, name).weight
        a, e = layer.self_attn, layer.mlp
        for name in ("q_norm", "k_norm"):
            direct[p + "self_attn." + name + ".weight"] = getattr(a, name).norm.weight
        for field in ("weight", "weight_scale_inv"):
            direct[p + "self_attn.o_proj." + field] = getattr(a.o_proj, field)
            for shard in ("q", "k", "v"):
                packed[p + "self_attn." + shard + "_proj." + field] = (getattr(a.qkv_proj, field), shard)
        direct[p + "mlp.gate.weight"] = e.gate_weight
        direct[p + "mlp.e_score_correction_bias"] = e.e_score_correction_bias
        for source, dest in (("gate_up_proj", "w13"), ("down_proj", "w2")):
            direct[p + "mlp.experts." + source] = getattr(e, dest)
            direct[p + "mlp.experts." + source + "_scale_inv"] = getattr(e, dest + "_weight_scale_inv")
    load_mapped_state(state_dict, direct, packed)
    prepare_linears(model)
