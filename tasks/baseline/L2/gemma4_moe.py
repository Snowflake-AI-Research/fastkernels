"""Gemma4 MoE block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gemma4_routing import Gemma4Routing
from ..L1.rms_norm import RMSNorm
from .fused_experts import FusedExperts

try:
    import vllm._custom_ops as vllm_ops
except ImportError:
    vllm_ops = None


class Gemma4GateLinear(nn.Module):
    """Replicated router projection with vLLM's bf16->fp32 router GEMM."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            vllm_ops is not None
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            return vllm_ops.router_gemm_bf16_fp32(x, self.weight)
        return F.linear(x.to(self.weight.dtype), self.weight).float()


class Gemma4Router(nn.Module):
    """Gemma4 router pre-normalization and expert-logit projection."""

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=False,
        )
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.root_size = config.hidden_size ** -0.5
        self.proj = Gemma4GateLinear(config.hidden_size, config.num_experts)

    def forward(self, x):
        x = self.norm(x)
        x = x * self.root_size
        x = x * self.scale.to(x.dtype)
        return self.proj(x)


class Gemma4MoE(nn.Module):
    """Gemma4 routed experts with full-softmax top-k routing."""

    def __init__(self, config):
        super().__init__()
        tp = _tp_size()
        self.tp_size = tp
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.intermediate_per_tp = config.moe_intermediate_size // tp

        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
        self.w13 = nn.Parameter(torch.empty(
            config.num_experts,
            2 * self.intermediate_per_tp,
            config.hidden_size,
        ))
        self.w2 = nn.Parameter(torch.empty(
            config.num_experts,
            config.hidden_size,
            self.intermediate_per_tp,
        ))
        self.w13.weight_loader = self._w13_weight_loader
        self.w2.weight_loader = self._w2_weight_loader

        self.routing = Gemma4Routing()
        self.fused_experts = FusedExperts(
            activation="gelu_tanh",
            config_style="vllm",
        )
        self.allreduce = AllReduce()
        self._use_custom_op = False
        self._layer_name = ""

    def _normalize_gate_up_weight(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.shape[1] == self.hidden_size:
            return loaded_weight.transpose(-1, -2).contiguous()
        return loaded_weight

    def _normalize_down_weight(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        if loaded_weight.shape[1] != self.hidden_size:
            return loaded_weight.transpose(-1, -2).contiguous()
        return loaded_weight

    def _w13_weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        weight = self._normalize_gate_up_weight(loaded_weight)
        full_inter = weight.shape[1] // 2
        n = self.intermediate_per_tp
        gate = weight[:, :full_inter, :]
        up = weight[:, full_inter:, :]
        param.data[:, :n, :].copy_(gate[:, rank * n:(rank + 1) * n, :])
        param.data[:, n:, :].copy_(up[:, rank * n:(rank + 1) * n, :])

    def _w2_weight_loader(self, param, loaded_weight):
        rank = _tp_rank()
        weight = self._normalize_down_weight(loaded_weight)
        n = self.intermediate_per_tp
        param.data.copy_(weight[:, :, rank * n:(rank + 1) * n])

    def forward_impl(self, hidden_states, router_logits):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        topk_weights, topk_ids = self.routing(
            router_logits.view(-1, self.num_experts),
            self.top_k,
            self.per_expert_scale,
        )
        out = self.fused_experts(
            hidden_states,
            self.w13,
            self.w2,
            topk_weights,
            topk_ids,
            self.num_experts,
        )
        if self.tp_size > 1:
            out = self.allreduce(out)
        return out.view(orig_shape)

    def forward(self, hidden_states, router_logits):
        if self._use_custom_op:
            return torch.ops.kb_nano.gemma4_moe_forward(
                hidden_states, router_logits, self._layer_name,
            )
        return self.forward_impl(hidden_states, router_logits)
