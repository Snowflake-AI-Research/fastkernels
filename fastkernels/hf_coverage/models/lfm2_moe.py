"""LFM2's short-convolution hybrid with its selected sigmoid-routed experts."""

import torch
from torch import nn

from fastkernels.hf_coverage.patches.lfm2_routing import Lfm2Routing
from fastkernels.tasks.baseline.L1.linear import Linear
from fastkernels.tasks.baseline.L2.fused_experts import FusedExperts

from .lfm2 import Lfm2ForCausalLM, make_workloads


class SparseExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.empty(
            config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size))
        self.experts.down_proj = nn.Parameter(torch.empty(
            config.num_experts, config.hidden_size, config.moe_intermediate_size))
        self.register_buffer("expert_bias", torch.empty(config.num_experts, dtype=torch.float32))
        self.router = Lfm2Routing(config.num_experts_per_tok, config.routed_scaling_factor)
        self.execute = FusedExperts()

    def forward(self, hidden):
        flat = hidden.reshape(-1, hidden.shape[-1])
        weights, indices = self.router(self.gate(flat), self.expert_bias)
        output = self.execute(flat, self.experts.gate_up_proj, self.experts.down_proj,
                              weights, indices, self.experts.gate_up_proj.shape[0])
        return output.reshape_as(hidden)


def build_from_config(config, device, dtype):
    if (not config.use_cache or config.rope_parameters["rope_type"] != "default"
            or not config.use_expert_bias or not config.norm_topk_prob):
        raise ValueError("The selected LFM2-MoE case uses cached default RoPE and normalized bias-corrected sigmoid routing")
    model = Lfm2ForCausalLM(config, config.intermediate_size)
    for index in range(config.num_dense_layers, config.num_hidden_layers):
        model.model.layers[index].feed_forward = SparseExperts(config)
    model = model.to(device=device, dtype=dtype).eval()
    for layer in model.model.layers[config.num_dense_layers:]:
        layer.feed_forward.expert_bias = layer.feed_forward.expert_bias.float()
    return model


def load_state_dict_into(model, weights, config):
    mapped = dict(weights)
    mapped["model.embed_tokens.emb.weight"] = mapped.pop("model.embed_tokens.weight")
    model.load_state_dict(mapped, strict=True)
