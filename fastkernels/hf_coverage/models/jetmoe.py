"""JetMoE routed query/output projections around shared-key/value attention."""

from copy import copy

import torch
from torch import nn

from fastkernels.hf_coverage.models.llama import make_workloads
from fastkernels.hf_coverage.models.olmo2 import decoder_config
from fastkernels.hf_coverage.patches.moe_sum_bias import BiasedMoeSum
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L1.moe_align import MoeAlign
from fastkernels.tasks.baseline.L1.moe_grouped_gemm import MoeGroupedGemm, get_triton_config
from fastkernels.tasks.baseline.L1.topk_softmax import TopKSoftmax
from fastkernels.tasks.baseline.L2.attention_impl import Attention
from fastkernels.tasks.baseline.L2.fused_experts import FusedExperts
from fastkernels.tasks.baseline.L4.llama import LlamaForCausalLM


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Linear(config.hidden_size, config.num_local_experts, False)
        self.top_k, self.topk = config.num_experts_per_tok, TopKSoftmax()

    def forward(self, hidden):
        # Renormalized full softmax equals softmax over the selected logits.
        return self.topk(self.gate(hidden).float(), self.top_k, renormalize=True)


class ExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)
        self.num_experts = config.num_local_experts
        self.w13 = nn.Parameter(torch.empty(self.num_experts, 2 * config.intermediate_size, config.hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, config.intermediate_size))
        self.experts = FusedExperts()
        self.experts.moe_sum = BiasedMoeSum(config.hidden_size)

    def forward(self, hidden):
        weights, indices = self.router(hidden)
        return self.experts(hidden, self.w13, self.w2, weights.to(hidden.dtype), indices, self.num_experts)


class AttentionExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)
        self.num_experts, self.top_k = config.num_local_experts, config.num_experts_per_tok
        self.width = config.kv_channels * config.num_key_value_heads
        self.input_weight = nn.Parameter(torch.empty(self.num_experts, self.width, config.hidden_size))
        self.output_weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, self.width))
        self.align, self.linear = MoeAlign(), MoeGroupedGemm()
        self.sum = BiasedMoeSum(config.hidden_size)

    def project(self, x, weight, indices, weights, top_k, weighted):
        n, out = x.shape[0], weight.shape[1]
        kernel = get_triton_config(n, weight.shape, (weight.shape[0], 0, out), top_k, use_fp8=False)
        tokens, experts, padded = self.align(indices, kernel["BLOCK_SIZE_M"], self.num_experts,
                                             naive=n * top_k * 4 <= self.num_experts)
        output = torch.empty(n * top_k, out, dtype=x.dtype, device=x.device)
        self.linear(x.contiguous(), weight, output, weights, tokens, experts, padded,
                    mul_routed_weight=weighted, top_k=top_k, config=kernel)
        return output

    def map(self, hidden):
        weights, indices = self.router(hidden)
        query = self.project(hidden, self.input_weight, indices, weights, self.top_k, False)
        return query.view(hidden.shape[0], self.top_k, self.width), (indices, weights)

    def reduce(self, hidden, routing):
        indices, weights = routing
        output = self.project(hidden.reshape(-1, self.width), self.output_weight,
                              indices.reshape(-1, 1), weights.reshape(-1, 1).to(hidden.dtype), 1, True)
        return self.sum(output, self.top_k)


class ExpertAttention(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        self.experts = AttentionExperts(config)
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.kv_channels
        self.top_k = config.num_experts_per_tok
        self.kv_proj = Linear(config.hidden_size, 2 * self.kv_heads * self.head_dim, False)
        self.rotary_emb = rotary
        self.attn = Attention(self.heads, self.head_dim, self.head_dim ** -0.5, num_kv_heads=self.kv_heads)

    def forward(self, positions, hidden):
        n = hidden.shape[0]
        query, routing = self.experts.map(hidden)
        key, value = self.kv_proj(hidden).chunk(2, -1)
        query, key = self.rotary_emb(positions, query.reshape(n, -1).contiguous(), key.contiguous())
        # HF repeats the whole KV-head group for each selected expert. The
        # paged operation groups adjacent query heads by KV head instead.
        query = query.view(n, self.top_k, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        output = self.attn(query.reshape(n, -1), key, value.contiguous())
        output = output.view(n, self.kv_heads, self.top_k, self.head_dim).transpose(1, 2).contiguous()
        return self.experts.reduce(output, routing)


def build_from_config(config, device, dtype):
    if config.activation_function != "silu" or not config.tie_word_embeddings or config.output_router_logits:
        raise ValueError("Selected JetMoE uses tied embeddings, SiLU and no router outputs")
    carrier = copy(config)
    carrier.head_dim = config.kv_channels
    model = LlamaForCausalLM(decoder_config(carrier, dtype))
    for layer in model.model.layers:
        # HF omits config.rms_norm_eps at these two call sites, retaining the
        # norm class's 1e-6 default. Its final norm uses the configured value.
        layer.input_layernorm.eps = 1e-6
        layer.post_attention_layernorm.eps = 1e-6
        layer.self_attn = ExpertAttention(config, model.model.rotary_emb)
        layer.mlp = ExpertMLP(config)
    model.lm_head.embedding_op.emb.weight = model.model.embed_tokens.embedding_op.emb.weight
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def load_state_dict_into(model, state_dict, config):
    if not torch.equal(state_dict["model.embed_tokens.weight"], state_dict["lm_head.weight"]):
        raise ValueError("JetMoE tied embedding values disagree")
    direct = {"model.embed_tokens.weight": model.model.embed_tokens.embedding_op.emb.weight,
              "model.norm.weight": model.model.norm.weight,
              "lm_head.weight": model.lm_head.embedding_op.emb.weight}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            direct[p + name + ".weight"] = getattr(layer, name).weight
        a = layer.self_attn
        direct[p + "self_attention.kv_proj.weight"] = a.kv_proj.weight
        for stem, expert in ((p + "self_attention.experts.", a.experts), (p + "mlp.", layer.mlp)):
            direct[stem + "router.layer.weight"] = expert.router.gate.weight
            attention = isinstance(expert, AttentionExperts)
            direct[stem + "input_linear.weight"] = expert.input_weight if attention else expert.w13
            direct[stem + "output_linear.weight"] = expert.output_weight if attention else expert.w2
            direct[stem + "bias"] = expert.sum.bias if attention else expert.experts.moe_sum.bias
    if state_dict.keys() != direct.keys():
        raise KeyError(f"JetMoE state mismatch: {sorted(state_dict.keys() ^ direct.keys())}")
    for name, parameter in direct.items():
        if parameter.shape != state_dict[name].shape or parameter.dtype != state_dict[name].dtype:
            raise ValueError(f"JetMoE weight mismatch: {name}")
        parameter.copy_(state_dict[name])
