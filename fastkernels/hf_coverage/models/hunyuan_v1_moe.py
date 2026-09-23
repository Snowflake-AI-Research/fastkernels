"""Hunyuan routed and shared experts with the existing dense attention construction."""

import torch

from fastkernels.hf_coverage.models.hunyuan_v1_dense import build_from_config as build_dense
from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.patches.hunyuan_moe_precision import HunyuanFusedExperts
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L2.parallel_linear import ReplicatedLinear


class FP32Gate(ReplicatedLinear):
    """The HF gate constructs its parameter explicitly in FP32, including BF16 loads."""

    def forward(self, hidden_states):
        return super().forward(hidden_states.float())


def build_from_config(config, device, dtype):
    model = build_dense(config, device, dtype)
    for i, layer in enumerate(model.model.layers):
        experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[i]
        topk = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[i]
        layer.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size, num_experts=experts, top_k=topk,
            moe_intermediate_size=config.intermediate_size,
            shared_expert_intermediate_size=config.intermediate_size,
            routing="softmax", renormalize=True, keep_router_weights_fp32=True,
        ).to(device=device, dtype=dtype)
        # Preserve the native rounded down projection and FP32 weighted reduction.
        # Select before loading, so expert weights retain their unshuffled layout.
        layer.mlp.use_trtllm = False
        layer.mlp.fused_experts = HunyuanFusedExperts()
        layer.mlp.gate = FP32Gate(config.hidden_size, experts, bias=False).to(device=device, dtype=torch.float32)
    return model.eval()


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
        direct[p + "self_attn.query_layernorm.weight"] = layer.self_attn.q_wl_norm.weight
        direct[p + "self_attn.key_layernorm.weight"] = layer.self_attn.k_wl_norm.weight
        for shard in ("q", "k", "v"):
            packed[p + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard)
        expert = layer.mlp
        direct[p + "mlp.gate.wg.weight"] = expert.gate.weight
        direct[p + "mlp.experts.gate_up_proj"] = expert.w13
        direct[p + "mlp.experts.down_proj"] = expert.w2
        direct[p + "mlp.shared_mlp.down_proj.weight"] = expert.shared_expert.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[p + f"mlp.shared_mlp.{name}.weight"] = (expert.shared_expert.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"Hunyuan MoE state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("Hunyuan MoE tied weights disagree")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"Hunyuan MoE weight shape/dtype mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
    for layer in model.model.layers:
        if layer.mlp.w13.dtype == torch.bfloat16:
            layer.mlp.process_weights_after_loading()
