"""Native block-FP8 DeepSeek V3 with compressed MLA and explicit HF routing."""

from dataclasses import fields

import torch

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.patches.grouped_topk_normalization import GroupedTopKNormalization
from fastkernels.tasks.baseline.L1.fp8_linear import Fp8Linear, postprocess_fp8_weights
from fastkernels.tasks.baseline.L1.linear import Matmul
from fastkernels.tasks.baseline.L2.deepseek_moe import DeepSeekMoE
from fastkernels.tasks.baseline.L2.vllm_fused_experts import VllmFusedExperts
from fastkernels.tasks.baseline.L4.deepseek import DeepSeekV3Config, DeepSeekV3ForCausalLM


class NativeFP8Experts(DeepSeekMoE):
    """Compose the existing FP8 experts with HF's FP32 routing and weight scale."""
    def __init__(self, config, quant_config, *, epsilon=1e-20, gate_fp32=True):
        super().__init__(config, quant_config)
        self.gate_fp32 = gate_fp32
        self.gate_matmul = Matmul()
        self.grouped_topk = GroupedTopKNormalization(scoring_func="sigmoid", epsilon=epsilon,
                                                    scale=self.routed_scaling_factor)
        self.fused_experts = VllmFusedExperts()

    def forward(self, hidden):
        if self.gate_fp32:
            scores = self.gate_matmul(hidden.float(), self.gate_weight.float())
        else:
            scores = self.gate_matmul(hidden.to(self.gate_weight.dtype), self.gate_weight).float()
        weights, indices = self.grouped_topk(scores, self.e_score_correction_bias,
                                            self.n_group, self.topk_group, self.top_k)
        output = self.fused_experts(hidden, self.w13, self.w2, weights, indices, self.num_experts,
                                   w13_scale=self.w13_weight_scale_inv, w2_scale=self.w2_weight_scale_inv,
                                   block_shape=[128, 128])
        return output + self.shared_expert(hidden) if self.shared_expert is not None else output


def prepare_linears(model):
    """Use each existing FP8 linear's required load-time weight preparation."""
    for module in model.modules():
        if isinstance(getattr(module, "linear_op", None), Fp8Linear):
            weight, scales = postprocess_fp8_weights(module.weight.data, module.weight_scale_inv.data)
            module.weight.data, module.weight_scale_inv.data = weight, scales


def build_from_config(config, device, dtype):
    if config.quantization_config["quant_method"] != "fp8" or config.q_lora_rank is None:
        raise ValueError("Selected DeepSeek V3 uses block FP8 and low-rank queries")
    values = {f.name: getattr(config, f.name) for f in fields(DeepSeekV3Config) if hasattr(config, f.name)}
    values.update(dtype=dtype, rope_theta=config.rope_parameters["rope_theta"], scoring_func="sigmoid",
                  moe_router_dtype="float32", kv_cache_dtype="auto")
    carrier = DeepSeekV3Config(**values)
    model = DeepSeekV3ForCausalLM(carrier, quant_config=config.quantization_config)
    for i, layer in enumerate(model.model.layers):
        if i >= config.first_k_dense_replace:
            layer.mlp = NativeFP8Experts(carrier, config.quantization_config)
    # FP8 weights and FP32 block scales must retain their native dtypes.
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
        a = layer.self_attn
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            direct[p + "self_attn." + name + ".weight"] = getattr(a, name).weight
        for name in ("q_b_proj", "kv_b_proj", "o_proj"):
            for field in ("weight", "weight_scale_inv"):
                direct[p + "self_attn." + name + "." + field] = getattr(getattr(a, name), field)
        for shard, name in enumerate(("q_a_proj", "kv_a_proj_with_mqa")):
            for field in ("weight", "weight_scale_inv"):
                packed[p + "self_attn." + name + "." + field] = (getattr(a.fused_qkv_a_proj, field), shard)
        if i < config.first_k_dense_replace:
            target, stem = layer.mlp, p + "mlp."
        else:
            e = layer.mlp
            direct[p + "mlp.gate.weight"] = e.gate_weight
            direct[p + "mlp.gate.e_score_correction_bias"] = e.e_score_correction_bias
            for source, dest in (("gate_up_proj", "w13"), ("down_proj", "w2")):
                direct[p + "mlp.experts." + source] = getattr(e, dest)
                direct[p + "mlp.experts." + source + "_scale_inv"] = getattr(e, dest + "_weight_scale_inv")
            target, stem = e.shared_expert, p + "mlp.shared_experts."
        for field in ("weight", "weight_scale_inv"):
            direct[stem + "down_proj." + field] = getattr(target.down_proj, field)
            for shard, name in enumerate(("gate_proj", "up_proj")):
                packed[stem + name + "." + field] = (getattr(target.gate_up_proj, field), shard)
    load_mapped_state(state_dict, direct, packed)
    prepare_linears(model)
    for layer in model.model.layers:
        layer.self_attn.finalize_absorbed_weights()


def load_mapped_state(state_dict, direct, packed):
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"FP8 state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape:
            raise ValueError(f"FP8 weight shape mismatch: {name}")
        parameter.data = state_dict[name].to(parameter.device).clone()
    for name, (parameter, shard) in packed.items():
        if parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"Packed FP8 dtype mismatch: {name}")
        parameter.weight_loader(parameter, state_dict[name], shard)
