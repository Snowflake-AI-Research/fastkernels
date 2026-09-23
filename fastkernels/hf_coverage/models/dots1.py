"""Dots1 dense/routed decoder with existing attention, expert and router operations."""

import torch

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.grouped_topk_normalization import GroupedTopKNormalization
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class NormalizedExperts(SharedExpertMoE):
    """Expose FP32 gate projection and HF's normalization before expert weighting."""

    def __init__(self, *, normalizer, **kwargs):
        super().__init__(correction_bias=True, use_grouped_topk=True, **kwargs)
        # The monolithic backend owns a different normalization; use the
        # parent's existing separate router/expert operations for this variant.
        self.use_trtllm = False
        self.gate_linear = Matmul()
        self.grouped_topk = normalizer

    def forward_impl(self, hidden_states):
        logits = self.gate_linear(hidden_states.float(), self.gate.weight.float())
        weights, indices = self._route(logits)
        if not self.keep_router_weights_fp32:
            weights = weights.to(hidden_states.dtype)
        output = self.fused_experts(hidden_states, self.w13, self.w2, weights, indices, self.num_experts)
        if self.has_shared_expert:
            output = output + self.shared_expert(hidden_states)
        return output


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.attention_bias or config.tie_word_embeddings:
        raise ValueError("Selected Dots1 uses bias-free SiLU and an untied head")
    if not config.norm_topk_prob or any(t != "full_attention" for t in config.layer_types):
        raise ValueError("Selected Dots1 uses normalized routing and full attention")
    model = LlamaForCausalLM(decoder_config(config, dtype))
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.q_norm = RMSNorm(model.config.head_dim, config.rms_norm_eps)
        layer.self_attn.k_norm = RMSNorm(model.config.head_dim, config.rms_norm_eps)
        if i >= config.first_k_dense_replace:
            layer.mlp = NormalizedExperts(
                hidden_size=config.hidden_size, num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                routing="sigmoid", keep_router_weights_fp32=True,
                num_expert_group=config.n_group, topk_group=config.topk_group,
                shared_expert_intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                normalizer=GroupedTopKNormalization(scoring_func="sigmoid", epsilon=1e-20,
                                                    scale=config.routed_scaling_factor),
            )
    return model.to(device=device, dtype=dtype).eval()


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
        direct[p + "self_attn.o_proj.weight"] = layer.self_attn.o_proj.weight
        for name in ("q_norm", "k_norm"):
            direct[p + f"self_attn.{name}.weight"] = getattr(layer.self_attn, name).weight
        for shard in ("q", "k", "v"):
            packed[p + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard)
        if i < config.first_k_dense_replace:
            target, prefix = layer.mlp, p + "mlp."
        else:
            expert = layer.mlp
            direct[p + "mlp.gate.weight"] = expert.gate.weight
            direct[p + "mlp.gate.e_score_correction_bias"] = expert.gate.e_score_correction_bias
            direct[p + "mlp.experts.gate_up_proj"] = expert.w13
            direct[p + "mlp.experts.down_proj"] = expert.w2
            target, prefix = expert.shared_expert, p + "mlp.shared_experts."
        direct[prefix + "down_proj.weight"] = target.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[prefix + name + ".weight"] = (target.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"Dots1 state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"Dots1 weight shape mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
