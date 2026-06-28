"""Jamba's MoE block (sparse MoE with softmax+top-k routing).

Reference: ``transformers.models.jamba.modeling_jamba.JambaSparseMoeBlock``
            and AI21's modeling code.

Differences from :class:`L2.mixtral_moe.MixtralMoE`:

  * Routing weights are NOT renormalised after top-k (Jamba uses raw
    softmax-then-topk weights).  Mixtral uses renormalised weights.
  * ``num_experts`` and ``num_experts_per_tok`` are read from ``config``
    using Jamba's field names (``num_experts`` / ``num_experts_per_tok``,
    not ``num_local_experts``).
  * No tensor parallelism: a single-GPU engine targets Jamba models that
    fit on a B200 (the open-weight Jamba-tiny and Jamba-v0.1 fit; the
    gated 1.5/1.7-Mini variants are 52 B but still fit).

Otherwise reuses the same fused Triton grouped-GEMM expert kernel
(:class:`L2.fused_experts.FusedExperts`), which is the SOTA path the
``MixtralMoE`` benchmark already validates against.

L1 ops: ``Linear`` (router), ``TopKSoftmax``, ``FusedExperts``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.topk_softmax import TopKSoftmax
from .fused_experts import FusedExperts


class JambaMoE(nn.Module):
    """Jamba sparse MoE FFN: router (Linear, no bias) -> softmax+top-k
    (no renormalise) -> fused expert grouped GEMM.

    Weight layout (one Parameter per expert pair, like MixtralMoE):
      ``router.weight``: [num_experts, hidden_size]
      ``w13``: [num_experts, 2*intermediate_size, hidden_size]
              -- gate (w1) and up (w3) stacked along dim 1.
      ``w2``:  [num_experts, hidden_size, intermediate_size]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.router = Linear(hidden_size, num_experts, bias=False)

        self.w13 = nn.Parameter(torch.empty(
            num_experts, 2 * intermediate_size, hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader

        self.w2 = nn.Parameter(torch.empty(
            num_experts, hidden_size, intermediate_size,
        ))
        self.w2.weight_loader = self._w2_weight_loader

        self.topk_softmax = TopKSoftmax()
        self.fused_experts = FusedExperts()

    # -----------------------------------------------------------------
    # Weight loaders.  Jamba's HF checkpoint stores per-expert
    # ``gate_proj``, ``up_proj``, ``down_proj`` matrices under
    # ``feed_forward.experts.{i}.{gate_proj|up_proj|down_proj}.weight``.
    # We pack them into the [E, 2*inter, hidden] / [E, hidden, inter]
    # parameters here.  No TP sharding (single-GPU engine).
    # -----------------------------------------------------------------
    def _w13_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        is_w1: bool,
    ) -> None:
        """``loaded_weight``: [intermediate_size, hidden_size] per expert."""
        N = loaded_weight.size(0)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(loaded_weight)

    def _w2_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ) -> None:
        """``loaded_weight``: [hidden_size, intermediate_size] per expert."""
        param.data[expert_id].copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size)

        router_logits = self.router(flat)
        # Jamba: softmax-then-topk with NO renormalisation of the chosen
        # weights (HF reference keeps raw softmax-then-topk values).
        topk_weights, topk_ids = self.topk_softmax(
            router_logits, self.top_k, renormalize=False,
        )
        topk_weights = topk_weights.to(flat.dtype)

        out = self.fused_experts(
            flat, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
        )
        return out.view(orig_shape)
