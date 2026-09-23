"""Laguna's variable-width local/global attention and gated expert decoder."""

from copy import copy
from math import ceil

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.log_sigmoid import LogSigmoid
from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding
from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding
from fastkernels.tasks.baseline.L2.attention import LlamaAttention
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class PartialRotary(nn.Module):
    def __init__(self, operation, head_dim, rotary_dim):
        super().__init__()
        self.operation, self.head_dim, self.rotary_dim = operation, head_dim, rotary_dim

    def forward(self, positions, q, k):
        n = q.shape[0]
        qh, kh = q.view(n, -1, self.head_dim), k.view(n, -1, self.head_dim)
        qr, kr = self.operation(positions, qh[..., :self.rotary_dim].contiguous(),
                                kh[..., :self.rotary_dim].contiguous())
        return (torch.cat((qr, qh[..., self.rotary_dim:]), -1).reshape_as(q),
                torch.cat((kr, kh[..., self.rotary_dim:]), -1).reshape_as(k))


class GatedAttention(LlamaAttention):
    def __init__(self, config, index, rotary):
        heads = config.num_attention_heads_per_layer[index]
        super().__init__(config.hidden_size, heads, config.num_key_value_heads, config.head_dim,
                         rotary_emb=rotary, qk_norm=True, rms_norm_eps=config.rms_norm_eps,
                         sliding_window=config.sliding_window if config.layer_types[index] == "sliding_attention" else None)
        self.g_proj = Linear(config.hidden_size, heads, False)
        self.log_sigmoid, self.product = LogSigmoid(), ProductGate()

    def forward(self, positions, hidden_states):
        n = hidden_states.shape[0]
        q_width, kv_width = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        q, k, v = self.qkv_proj(hidden_states).split((q_width, kv_width, kv_width), -1)
        q = self.q_norm(q.view(n, self.num_heads, self.head_dim)).reshape(n, q_width)
        k = self.k_norm(k.view(n, self.num_kv_heads, self.head_dim)).reshape(n, kv_width)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
        # softplus(x) = -log_sigmoid(-x), using the existing stable operation.
        gate = -self.log_sigmoid(-self.g_proj(hidden_states).float())
        gate = gate.to(output.dtype).unsqueeze(-1).expand(n, self.num_heads, self.head_dim).reshape(n, q_width)
        return self.o_proj(self.product(torch.cat((output, gate), -1)))


class LagunaExperts(SharedExpertMoE):
    def __init__(self, config):
        super().__init__(hidden_size=config.hidden_size, num_experts=config.num_experts,
                         top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                         shared_expert_intermediate_size=config.shared_expert_intermediate_size,
                         routing="sigmoid", correction_bias=True)
        # HF rounds its projection before the sigmoid and its normalized
        # weights before expert weighting; use the existing separate path.
        self.use_trtllm = False
        self.scale = config.moe_routed_scaling_factor

    def forward_impl(self, hidden_states):
        weights, indices = self._route(self.gate(hidden_states).float())
        output = self.fused_experts(hidden_states, self.w13, self.w2, weights.to(hidden_states.dtype),
                                    indices, self.num_experts)
        return output * self.scale + self.shared_expert(hidden_states)


def build_from_config(config, device, dtype):
    if config.moe_router_logit_softcapping or config.moe_apply_router_weight_on_input or config.attention_bias:
        raise ValueError("Selected Laguna uses uncapped routing, output expert weighting and bias-free attention")
    carrier = copy(config)
    carrier.rope_parameters = config.rope_parameters["sliding_attention"]
    model = LlamaForCausalLM(decoder_config(carrier, dtype))
    rotary = {}
    for kind in set(config.layer_types):
        rope = config.rope_parameters[kind]
        width = int(config.head_dim * rope["partial_rotary_factor"])
        if rope["rope_type"] == "yarn":
            # This operation multiplies its capacity argument by the factor;
            # HF's configured maximum already includes that expansion.
            capacity = ceil(config.max_position_embeddings / rope["factor"])
            operation = YaRNRotaryEmbedding(width, capacity, rope["rope_theta"],
                                            rope["factor"], rope["original_max_position_embeddings"],
                                            beta_fast=rope["beta_fast"], beta_slow=rope["beta_slow"])
        elif rope["rope_type"] == "default":
            operation = RotaryEmbedding(width, config.max_position_embeddings, rope["rope_theta"])
        else:
            raise ValueError(f"Unexpected Laguna RoPE type: {rope['rope_type']}")
        rotary[kind] = PartialRotary(operation, config.head_dim, width)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = GatedAttention(config, i, rotary[config.layer_types[i]])
        if config.mlp_layer_types[i] == "sparse":
            layer.mlp = LagunaExperts(config)
    # Each attention type owns its corresponding rotary operation.
    del model.model.rotary_emb
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
        for name in ("o_proj", "g_proj", "q_norm", "k_norm"):
            direct[p + f"self_attn.{name}.weight"] = getattr(layer.self_attn, name).weight
        for shard in ("q", "k", "v"):
            packed[p + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard)
        if config.mlp_layer_types[i] == "dense":
            target, stem = layer.mlp, p + "mlp."
        else:
            expert = layer.mlp
            direct[p + "mlp.gate.weight"] = expert.gate.weight
            direct[p + "mlp.gate.e_score_correction_bias"] = expert.gate.e_score_correction_bias
            direct[p + "mlp.experts.gate_up_proj"] = expert.w13
            direct[p + "mlp.experts.down_proj"] = expert.w2
            target, stem = expert.shared_expert, p + "mlp.shared_experts."
        direct[stem + "down_proj.weight"] = target.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[stem + name + ".weight"] = (target.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"Laguna state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"Laguna weight mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
