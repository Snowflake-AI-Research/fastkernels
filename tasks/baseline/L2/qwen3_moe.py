"""Qwen3 Mixture-of-Experts block with fused Triton grouped GEMM.

Supports FP8 W8A8 block-scaled expert weights. When quant_config is set,
expert weights are stored as float8_e4m3fn with per-block scale factors,
and the fused experts kernel runs FP8 dot products with FP32 accumulation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.topk_softmax import TopKSoftmax
from ..L2.fused_experts import FusedExperts
from ..L2.parallel_linear import ReplicatedLinear

_FP8_BLOCK = 128


class Qwen3MoE(nn.Module):
    """Qwen3 Mixture-of-Experts with fused Triton grouped GEMM.

    Weights (FP8 mode):
      gate:     [num_experts, hidden_size] (bfloat16, replicated)
      w13:      [E, 2*moe_intermediate_per_tp, hidden_size] (float8_e4m3fn)
      w13_scale:[E, scale_rows_13, scale_cols_13] (float32)
      w2:       [E, hidden_size, moe_intermediate_per_tp] (float8_e4m3fn)
      w2_scale: [E, scale_rows_2, scale_cols_2] (float32)

    Weights (BF16 mode):
      gate:  [num_experts, hidden_size]
      w13:   [E, 2*moe_intermediate_per_tp, hidden_size]
      w2:    [E, hidden_size, moe_intermediate_per_tp]
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.use_fp8 = quant_config is not None

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False,
        )

        w13_rows = 2 * self.intermediate_per_tp
        w2_cols = self.intermediate_per_tp

        if self.use_fp8:
            block_size = quant_config.get("weight_block_size", [128, 128])
            self.block_shape = block_size
            block_n, block_k = block_size[0], block_size[1]

            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w13_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(w13_rows / block_n),
                math.ceil(config.hidden_size / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w2_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(config.hidden_size / block_n),
                math.ceil(w2_cols / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w13.weight_loader = self._w13_weight_loader_fp8
            self.w13_scale.weight_loader = self._w13_scale_loader
            self.w2.weight_loader = self._w2_weight_loader_fp8
            self.w2_scale.weight_loader = self._w2_scale_loader
        else:
            self.block_shape = None
            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
            ))
            self.w13.weight_loader = self._w13_weight_loader

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
            ))
            self.w2.weight_loader = self._w2_weight_loader

            self.w13_scale = None
            self.w2_scale = None

        self.topk_softmax = TopKSoftmax()
        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

        # Custom-op dispatch for torch.compile (set by engine after model init)
        self._use_custom_op = False
        self._layer_name = ""

    # --- BF16 weight loaders ---

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    # --- FP8 weight loaders ---

    def _w13_weight_loader_fp8(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        block_n = self.block_shape[0]
        N = self.intermediate_per_tp
        scale_rows_per_shard = math.ceil(N / block_n)
        full_scale_rows = loaded_weight.shape[0]
        rows_per_tp = full_scale_rows // tp
        src = loaded_weight.narrow(0, rank * rows_per_tp, rows_per_tp)
        offset = 0 if is_w1 else scale_rows_per_shard
        param.data[expert_id, offset:offset + rows_per_tp, :].copy_(src)

    def _w2_weight_loader_fp8(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        block_k = self.block_shape[1]
        N = self.intermediate_per_tp
        scale_cols_per_shard = math.ceil(N / block_k)
        full_scale_cols = loaded_weight.shape[1]
        cols_per_tp = full_scale_cols // tp
        src = loaded_weight.narrow(1, rank * cols_per_tp, cols_per_tp)
        param.data[expert_id].copy_(src)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core MoE logic, callable from both eager and custom-op paths."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self.topk_softmax(
            router_logits, self.top_k, renormalize=self.renormalize,
        )

        w13_scale_dg = getattr(self, 'w13_scale_dg', None)
        w2_scale_dg = getattr(self, 'w2_scale_dg', None)

        out = self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
            w13_scale=self.w13_scale,
            w2_scale=self.w2_scale,
            w13_scale_dg=w13_scale_dg,
            w2_scale_dg=w2_scale_dg,
            use_fp8_w8a8=self.use_fp8,
            block_shape=self.block_shape,
        )

        if self.tp_size > 1:
            out = self.allreduce(out)

        return out.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            return torch.ops.kb_nano.moe_forward(hidden_states, self._layer_name)
        return self.forward_impl(hidden_states)
