"""AFMoE dual normalization, gated local/global attention, and shared experts."""

import math

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import PostNormModel, decoder_config
from fastkernels.hf_coverage.patches.product_gate import ProductGate
from fastkernels.tasks.baseline.L1.bitnet_rms_norm import BitNetRMSNorm
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class GatedAttention(nn.Module):
    def __init__(self, parent, config, index):
        super().__init__()
        self.qkv_proj, self.o_proj, self.attn = parent.qkv_proj, parent.o_proj, parent.attn
        self.rotary_emb = parent.rotary_emb if config.layer_types[index] == "sliding_attention" else None
        self.attn = Attention(parent.num_heads, parent.head_dim, parent.head_dim ** -0.5,
                              num_kv_heads=parent.num_kv_heads,
                              sliding_window=config.sliding_window if self.rotary_emb is not None else None)
        self.num_heads, self.num_kv_heads, self.head_dim = parent.num_heads, parent.num_kv_heads, parent.head_dim
        self.q_norm = BitNetRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = BitNetRMSNorm(self.head_dim, config.rms_norm_eps)
        self.gate_proj = Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.sigmoid, self.product = Sigmoid(), ProductGate()

    def forward(self, positions, hidden_states):
        n = hidden_states.shape[0]
        q_width, kv_width = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        q, k, v = self.qkv_proj(hidden_states).split([q_width, kv_width, kv_width], dim=-1)
        q = self.q_norm(q.reshape(n, self.num_heads, self.head_dim).contiguous()).reshape(n, q_width)
        k = self.k_norm(k.reshape(n, self.num_kv_heads, self.head_dim).contiguous()).reshape(n, kv_width)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
        gate = self.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(self.product(torch.cat((output, gate), dim=-1)))


class AfExperts(SharedExpertMoE):
    def __init__(self, config):
        super().__init__(hidden_size=config.hidden_size, num_experts=config.num_experts,
                         top_k=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size,
                         shared_expert_intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                         routing="sigmoid", correction_bias=True, keep_router_weights_fp32=True)
        self.use_trtllm = False
        self.route_scale = config.route_scale

    def forward_impl(self, hidden_states):
        weights, indices = self._route(self.gate(hidden_states).float())
        weights = weights * self.route_scale
        output = self.fused_experts(hidden_states, self.w13, self.w2, weights, indices, self.num_experts)
        return self.shared_expert(hidden_states) + output


class DualNormLayer(nn.Module):
    def __init__(self, parent, config, index):
        super().__init__()
        self.self_attn = GatedAttention(parent.self_attn, config, index)
        self.mlp = parent.mlp if index < config.num_dense_layers else AfExperts(config)
        for name in ("input_layernorm", "post_attention_layernorm", "pre_mlp_layernorm", "post_mlp_layernorm"):
            setattr(self, name, BitNetRMSNorm(config.hidden_size, config.rms_norm_eps))

    def forward(self, positions, hidden_states, residual=None):
        hidden_states = hidden_states + self.post_attention_layernorm(
            self.self_attn(positions, self.input_layernorm(hidden_states)))
        hidden_states = hidden_states + self.post_mlp_layernorm(self.mlp(self.pre_mlp_layernorm(hidden_states)))
        return hidden_states, None


class AfModel(PostNormModel):
    def forward(self, input_ids, positions, inputs_embeds=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        return super().forward(input_ids, positions, inputs_embeds=hidden * self.embedding_scale)


def build_from_config(config, device, dtype):
    if config.hidden_act != "silu" or config.tie_word_embeddings or config.rope_parameters["rope_type"] != "default":
        raise ValueError("Selected AFMoE uses untied SiLU/default-RoPE computation")
    fk = decoder_config(config, dtype)
    model = LlamaForCausalLM(fk)
    model.model = AfModel(fk)
    model.model.embedding_scale = math.sqrt(config.hidden_size) if config.mup_enabled else 1.0
    model.model.layers = nn.ModuleList(DualNormLayer(layer, config, i) for i, layer in enumerate(model.model.layers))
    model.model.norm = BitNetRMSNorm(config.hidden_size, config.rms_norm_eps)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    packed = {}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm", "pre_mlp_layernorm", "post_mlp_layernorm"):
            direct[p + name + ".weight"] = getattr(layer, name).weight
        for name in ("o_proj", "gate_proj", "q_norm", "k_norm"):
            direct[p + f"self_attn.{name}.weight"] = getattr(layer.self_attn, name).weight
        for shard in ("q", "k", "v"):
            packed[p + f"self_attn.{shard}_proj.weight"] = (layer.self_attn.qkv_proj.weight, shard)
        if i < config.num_dense_layers:
            target, prefix = layer.mlp, p + "mlp."
        else:
            expert = layer.mlp
            direct[p + "mlp.router.gate.weight"] = expert.gate.weight
            direct[p + "mlp.expert_bias"] = expert.gate.e_score_correction_bias
            direct[p + "mlp.experts.gate_up_proj"] = expert.w13
            direct[p + "mlp.experts.down_proj"] = expert.w2
            target, prefix = expert.shared_expert, p + "mlp.shared_experts."
        direct[prefix + "down_proj.weight"] = target.down_proj.weight
        for shard, name in enumerate(("gate_proj", "up_proj")):
            packed[prefix + name + ".weight"] = (target.gate_up_proj.weight, shard)
    if state_dict.keys() != direct.keys() | packed.keys():
        raise KeyError(f"AFMoE state mismatch: {sorted(state_dict.keys() ^ (direct.keys() | packed.keys()))}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"AFMoE weight shape/dtype mismatch: {name}")
        parameter.copy_(state_dict[name])
    for name, (parameter, shard) in packed.items():
        parameter.weight_loader(parameter, state_dict[name], shard)
