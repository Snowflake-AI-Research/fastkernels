"""DeepSeek V2 Lite with direct queries and HF's expanded attention representation."""

from copy import copy

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads as llama_workloads
from fastkernels.hf_coverage.runner import Workload
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.ernie4_5_rope import FP32RotaryEmbedding
from fastkernels.tasks.baseline.L1.linear import Linear, Matmul
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YarnRotaryEmbedding
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class ExpandedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.nope, self.rope = config.qk_nope_head_dim, config.qk_rope_head_dim
        self.value, self.rank = config.v_head_dim, config.kv_lora_rank
        width = self.nope + self.rope
        self.q_proj = Linear(config.hidden_size, self.heads * width, False)
        self.kv_a_proj_with_mqa = Linear(config.hidden_size, self.rank + self.rope, False)
        self.kv_a_layernorm = RMSNorm(self.rank, 1e-6)
        self.kv_b_proj = Linear(self.rank, self.heads * (self.nope + self.value), False)
        self.o_proj = Linear(self.heads * self.value, config.hidden_size, False)
        # The default HND cache store assumes a power-of-two head width;
        # the existing NHD/Triton path supports the native 192-wide Q/K.
        self.attn = Attention(self.heads, width, width ** -0.5, num_kv_heads=self.heads,
                              prefer_triton=True)

    def forward(self, positions, hidden_states):
        count = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(count, self.heads, self.nope + self.rope)
        q_plain, q_rotary = q.split((self.nope, self.rope), -1)
        latent, k_rotary = self.kv_a_proj_with_mqa(hidden_states).split((self.rank, self.rope), -1)
        kv = self.kv_b_proj(self.kv_a_layernorm(latent.contiguous()))
        k_plain, value = kv.view(count, self.heads, self.nope + self.value).split((self.nope, self.value), -1)
        q_rotary, k_rotary = self.rotary_emb(positions, q_rotary.contiguous(), k_rotary.contiguous())
        q = torch.cat((q_plain, q_rotary), -1)
        k = torch.cat((k_plain, k_rotary.view(count, 1, self.rope).expand(-1, self.heads, -1)), -1)
        # The existing paged kernel requires equal Q/K/V head widths. This is
        # the same constant padding and output slice as HF's FlashAttention path.
        value = torch.nn.functional.pad(value, (0, self.nope + self.rope - self.value))
        output = self.attn(q, k, value).view(count, self.heads, -1)[..., :self.value]
        return self.o_proj(output.reshape(count, -1).contiguous())


class DirectRouterExperts(SharedExpertMoE):
    def __init__(self, **kwargs):
        super().__init__(use_grouped_topk=True, keep_router_weights_fp32=True, **kwargs)
        self.use_trtllm = False
        self.gate_linear = Matmul()

    def forward_impl(self, hidden_states):
        scores = self.gate_linear(hidden_states.float(), self.gate.weight.float())
        weights, indices = self._route(scores)
        weights = weights * self.routed_scaling_factor
        output = self.fused_experts(hidden_states, self.w13, self.w2, weights, indices, self.num_experts)
        return output + self.shared_expert(hidden_states) if self.has_shared_expert else output


def build_from_config(config, device, dtype):
    if config.q_lora_rank is not None or config.topk_method != "greedy" or config.attention_bias or config.mlp_bias:
        raise ValueError("Selected DeepSeek V2 Lite uses direct queries, greedy routing and bias-free projections")
    carrier = copy(config)
    carrier.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    carrier.num_key_value_heads = config.num_attention_heads
    model = LlamaForCausalLM(decoder_config(carrier, dtype))
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = ExpandedAttention(config)
        if i >= config.first_k_dense_replace:
            layer.mlp = DirectRouterExperts(
                hidden_size=config.hidden_size, num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                renormalize=False, routed_scaling_factor=config.routed_scaling_factor,
                shared_expert_intermediate_size=config.n_shared_experts * config.moe_intermediate_size,
            )
    model.to(device=device, dtype=dtype)
    rope = config.rope_parameters
    yarn = YarnRotaryEmbedding(config.qk_rope_head_dim, rope["original_max_position_embeddings"],
                              rope["rope_theta"], rope["factor"], beta_fast=rope["beta_fast"],
                              beta_slow=rope["beta_slow"], mscale=rope["mscale"],
                              mscale_all_dim=rope["mscale_all_dim"]).to(device=device)
    # Reuse the existing precision patch with the unchanged YaRN cache.
    rotary = FP32RotaryEmbedding(config.qk_rope_head_dim, 1, rope["rope_theta"],
                                is_neox_style=False).to(device=device)
    rotary.cos_sin_cache = yarn.cos_sin_cache
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = rotary
    return model.eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            direct[prefix + name + ".weight"] = getattr(layer, name).weight
        for name in ("q_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj"):
            direct[prefix + "self_attn." + name + ".weight"] = getattr(layer.self_attn, name).weight
        if i < config.first_k_dense_replace:
            target, stem = layer.mlp, prefix + "mlp."
        else:
            expert = layer.mlp
            direct[prefix + "mlp.gate.weight"] = expert.gate.weight
            direct[prefix + "mlp.experts.gate_up_proj"] = expert.w13
            direct[prefix + "mlp.experts.down_proj"] = expert.w2
            target, stem = expert.shared_expert, prefix + "mlp.shared_experts."
        direct[stem + "down_proj.weight"] = target.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[stem + name + ".weight"] = (target.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"DeepSeek V2 state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"DeepSeek V2 weight mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)


def make_workloads(model, inputs, config, *, case=None):
    workloads = llama_workloads(model, inputs, config, case=case)
    if case is None or case.get("workload") != "causal_lm_continuation":
        return workloads
    for name, work in list(workloads.items()):
        def collect(output, parent=work.collect):
            output = parent(output)
            # Physical value pages include constant head-width padding. Compare
            # only HF's logical values; the full pages remain in timed execution.
            return {key: value[..., :config.v_head_dim] if key.endswith(".value") else value
                    for key, value in output.items()}
        workloads[name] = Workload(run=work.run, prepare=work.prepare, collect=collect)
    return workloads
