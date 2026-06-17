from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gate_linear import GateLinear
from ..L1.grouped_topk import GroupedTopK
from .fused_experts import FusedExperts
from .llama_mlp import LlamaMLP
from .parallel_linear import ReplicatedLinear


class KimiMoE(nn.Module):
    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.num_shared_experts = config.num_shared_experts
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tp_size = _tp_size()
        self.intermediate_per_tp = config.moe_intermediate_size // self.tp_size

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32),
        )
        self.gate.e_score_correction_bias.weight_loader = (
            lambda p, w: p.data.copy_(w.to(p.dtype))
        )

        self.grouped_topk = GroupedTopK(
            scoring_func=config.moe_router_activation_func,
            renormalize=config.moe_renormalize,
            routed_scaling_factor=1.0,
            force_sorted=True,
        )
        self.w13 = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.intermediate_per_tp,
                config.hidden_size,
            ),
        )
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                self.intermediate_per_tp,
            ),
        )
        self.w2.weight_loader = self._w2_weight_loader
        self.fused_experts = FusedExperts()
        self.gate_linear = GateLinear()
        self.shared_experts = (
            LlamaMLP(
                config,
                quant_config=quant_config,
                intermediate_size=config.moe_intermediate_size * self.num_shared_experts,
                reduce_results=False,
            )
            if self.num_shared_experts
            else None
        )
        self.allreduce = AllReduce()

        # Custom-op dispatch for torch.compile (flipped by enable_custom_ops
        # once the model is wrapped with torch.compile). ``_layer_name`` is
        # populated by auto_register_no_compile_layers.
        self._use_custom_op = False
        self._layer_name = ""

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        n = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, _tp_rank() * n, n)
        offset = 0 if is_w1 else n
        param.data[expert_id, offset:offset + n, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        n = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, _tp_rank() * n, n))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            return torch.ops.kb_nano.moe_forward(hidden_states, self._layer_name)
        return self.forward_impl(hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )

        router_logits = self.gate_linear(
            hidden_states,
            self.gate.weight,
            out_dtype=torch.float32,
        )
        topk_weights, topk_ids = self.grouped_topk(
            router_logits,
            self.gate.e_score_correction_bias,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            topk=self.top_k,
        )

        out = self.fused_experts(
            hidden_states,
            self.w13,
            self.w2,
            topk_weights,
            topk_ids,
            self.num_experts,
        )
        out = out * self.routed_scaling_factor
        if shared_output is not None:
            out = out + shared_output
        if self.tp_size > 1:
            out = self.allreduce(out)
        return out.view(orig_shape)
