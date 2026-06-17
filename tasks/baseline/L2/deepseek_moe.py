"""DeepSeek MoE with shared expert, grouped routing, and FP8 expert execution.

Uses :class:`GroupedTopK` for routing, :class:`VllmFusedExperts` for the
FP8 expert path (a fresh-allocation port of vLLM's ``fused_experts_impl``
that mirrors vLLM's Triton oracle for Hopper + block-FP8 + TP),
:class:`FusedExperts` for the BF16 / unquantized fallback, and a shared
expert (``LlamaMLP`` with ``reduce_results=False``) that runs on a
separate CUDA stream for overlap.

Matches vllm's DeepseekV2MoE: routed_scaling_factor is applied post-experts
(not folded into routing weights), and the shared expert uses
``moe_intermediate_size * n_shared_experts`` as its intermediate dimension.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.grouped_topk import GroupedTopK
from ..L1.gate_linear import GateLinear
from .fused_experts import FusedExperts
from .llama_mlp import LlamaMLP
from .vllm_fused_experts import VllmFusedExperts

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


class DeepSeekMoE(nn.Module):
    """DeepSeek Mixture-of-Experts with shared expert and grouped routing.

    Architecture:
    - Router: replicated gate + e_score_correction_bias
    - Shared expert: LlamaMLP (reduce_results=False)
    - Routed experts: FP8 weights (w13, w2) + per-block scales
    - Routing: GroupedTopK (sigmoid + grouped top-k with bias, matches vLLM)
    - routed_scaling_factor applied post-experts (not in routing weights)
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        self.use_fp8 = quant_config is not None

        n_group = getattr(config, 'n_group', 1)
        topk_group = getattr(config, 'topk_group', 1)
        self.n_group = n_group
        self.topk_group = topk_group
        self.scoring_func = getattr(config, 'scoring_func', 'softmax')
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.topk_method = getattr(config, 'topk_method', 'noaux_tc')

        self.gate_weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size),
        )
        self.gate_weight.weight_loader = lambda p, w: p.data.copy_(w)

        # ``e_score_correction_bias`` only exists for the ``noaux_tc`` topk
        # method (matches vLLM's ``DeepseekV2MoE`` which only allocates it
        # when ``config.topk_method == "noaux_tc"``). For other methods
        # (e.g., ``greedy``), the bias is simply absent.
        if self.topk_method == 'noaux_tc':
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(config.n_routed_experts, dtype=torch.float32),
            )
            self.e_score_correction_bias.weight_loader = (
                lambda p, w: p.data.copy_(w)
            )
        else:
            self.register_parameter('e_score_correction_bias', None)

        n_shared = getattr(config, 'n_shared_experts', 1)
        if n_shared is not None and n_shared > 0:
            shared_intermediate = config.moe_intermediate_size * n_shared
            # LlamaMLP with reduce_results=False — the final all-reduce
            # is deferred and runs after routed + shared expert are summed.
            self.shared_expert = LlamaMLP(
                config,
                quant_config=quant_config,
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate,
                reduce_results=False,
            )
        else:
            self.shared_expert = None

        if self.use_fp8:
            self.w13 = nn.Parameter(torch.empty(
                config.n_routed_experts, 2 * self.intermediate_per_tp, config.hidden_size,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w2 = nn.Parameter(torch.empty(
                config.n_routed_experts, config.hidden_size, self.intermediate_per_tp,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w13_weight_scale_inv = nn.Parameter(torch.empty(
                config.n_routed_experts,
                *_scale_shape(2 * self.intermediate_per_tp, config.hidden_size),
                dtype=torch.float32,
            ), requires_grad=False)
            self.w2_weight_scale_inv = nn.Parameter(torch.empty(
                config.n_routed_experts,
                *_scale_shape(config.hidden_size, self.intermediate_per_tp),
                dtype=torch.float32,
            ), requires_grad=False)
        else:
            self.w13 = nn.Parameter(torch.empty(
                config.n_routed_experts, 2 * self.intermediate_per_tp, config.hidden_size,
            ))
            self.w2 = nn.Parameter(torch.empty(
                config.n_routed_experts, config.hidden_size, self.intermediate_per_tp,
            ))

        self.w13.weight_loader = self._w13_weight_loader
        self.w2.weight_loader = self._w2_weight_loader
        if self.use_fp8:
            self.w13_weight_scale_inv.weight_loader = self._w13_scale_loader
            self.w2_weight_scale_inv.weight_loader = self._w2_scale_loader

        # Routing weights: vLLM always passes ``routed_scaling_factor=1.0``
        # to ``grouped_topk`` and applies the factor *post-experts* (see
        # ``vllm/model_executor/models/deepseek_v2.py:325`` and L378-379).
        # We mirror that by leaving the factor at 1.0 here.
        self.grouped_topk = GroupedTopK(
            scoring_func=self.scoring_func,
            renormalize=self.norm_topk_prob,
            routed_scaling_factor=1.0,
        )
        self.gate = GateLinear()
        # FP8 path uses a fresh-allocation, vLLM-mirrored op so it is
        # both bit-identical to vLLM's Triton MoE *and* safe to compose
        # with CUDA graph capture (no shared scratch buffers that an
        # eager prefill could reallocate underneath a captured graph).
        # The BF16 / unquantized path keeps the standard ``FusedExperts``.
        if self.use_fp8:
            self.fused_experts = VllmFusedExperts()
        else:
            self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()
        # Mirrors vLLM's ``VLLM_DISABLE_SHARED_EXPERTS_STREAM`` env knob
        # (default: stream enabled). When disabled, the shared expert runs
        # serially on the main stream, which is helpful for debugging and
        # for arches where the secondary stream is harmful.
        import os as _os
        self._disable_shared_stream: bool = (
            _os.environ.get("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0") != "0"
        )
        self._shared_stream: torch.cuda.Stream | None = None

        # Custom-op dispatch scaffolding (matches the other MoE L2 modules).
        # ``_use_custom_op`` is flipped to True by ``enable_custom_ops`` so
        # ``torch.compile`` sees the MoE block as an opaque
        # ``kb_nano::moe_forward`` op (avoids tracing into CUDA stream and
        # DeepGEMM pybind boundaries).
        self._use_custom_op = False
        self._layer_name = ""

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

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        scale_rows = math.ceil(N / _FP8_BLOCK)
        offset = 0 if is_w1 else scale_rows
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data[expert_id, offset:offset + scale_rows, :].copy_(src)

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        scale_cols = math.ceil(N / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 1)[rank]
        param.data[expert_id].copy_(src)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            return torch.ops.kb_nano.moe_forward(hidden_states, self._layer_name)
        return self.forward_impl(hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Launch shared expert on a separate CUDA stream for overlap.
        # Matches vLLM's monolithic DeepSeekV2MoE behaviour
        # (``DefaultMoeRunner`` in
        # ``vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py``).
        shared_out = None
        use_shared_stream = (
            self.shared_expert is not None and not self._disable_shared_stream
        )
        if use_shared_stream:
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream()
            self._shared_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._shared_stream):
                shared_out = self.shared_expert(hidden_states)
        elif self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)

        # Router gate with vLLM-parity dispatch. vLLM's GateLinear routes
        # through three tiers (DSV3 specialised kernel → cuBLAS BF16→FP32 →
        # PyTorch F.linear) and the accumulation order matters: a "promote
        # both to FP32 then matmul" path produces a different bit pattern
        # that flips near-tie group/expert selection in the noaux_tc
        # grouped-topk path.
        #
        # vLLM chooses the router ``out_dtype`` in
        # ``deepseek_v2.py:DeepseekV2MoE.__init__``:
        #
        #     self.gate.set_out_dtype(
        #         torch.float32
        #         if self.experts.quant_method.is_monolithic
        #         and self.experts.routing_method_type ==
        #             RoutingMethodType.DeepSeekV3
        #         else torch.bfloat16
        #     )
        #
        # For FP8 blockwise DeepSeek-V3 experts (our ``use_fp8`` path) the
        # quant method is *not* ``is_monolithic``, so vLLM uses BF16 here
        # — and the router_logits carry BF16 precision, which matters for
        # near-tie top-k boundaries. For the BF16 / "unquantized" path vLLM
        # is monolithic, so it keeps FP32. We mirror that choice exactly.
        router_out_dtype = (
            torch.float32 if not self.use_fp8 else torch.bfloat16
        )
        router_logits = self.gate(
            hidden_states, self.gate_weight, out_dtype=router_out_dtype,
        )
        topk_weights, topk_ids = self.grouped_topk(
            router_logits, self.e_score_correction_bias,
            self.n_group, self.topk_group, self.top_k,
        )
        # ``topk_weights`` is FP32 (matches vLLM). For the FP8 expert path
        # we pass FP32 weights through verbatim so the Triton kernel reads
        # them at the same precision vLLM does (vLLM's grouped_topk router
        # returns FP32 — see
        # ``vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:165``
        # — and ``invoke_fused_moe_triton_kernel`` consumes them directly).
        # Cast to activation dtype only for the BF16 / "unquantized" path,
        # which still uses kb_nano's local FusedExperts wrapper.
        if self.use_fp8:
            # FP8 W8A8 block-quant on Hopper + TP: vLLM's oracle
            # (``select_fp8_moe_backend`` -> ``_get_priority_backends``
            # in ``vllm/.../fused_moe/oracle/fp8.py``) explicitly moves
            # ``TRITON`` to the front for this configuration.  The
            # DeepGEMM path drifts ~1 BF16 ULP which cascades through
            # near-tie expert selection in subsequent MoE layers.
            # ``VllmFusedExperts`` is a fresh-allocation port of vLLM's
            # ``fused_experts_impl`` Triton path and consumes
            # ``topk_weights`` in FP32 (vLLM's ``GroupedTopKRouter``
            # returns FP32 and the Triton kernel scales in FP32 before
            # the final ``.to(compute_type)`` cast).
            out = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
                w13_scale=self.w13_weight_scale_inv,
                w2_scale=self.w2_weight_scale_inv,
                block_shape=[_FP8_BLOCK, _FP8_BLOCK],
            )
        else:
            topk_weights_act = topk_weights.to(hidden_states.dtype)
            out = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights_act, topk_ids, self.num_experts,
            )

        # ``routed_scaling_factor`` is applied *post-experts*, matching
        # ``vllm/model_executor/models/deepseek_v2.py:378-379``.
        out = out * self.routed_scaling_factor

        if shared_out is not None:
            if use_shared_stream:
                torch.cuda.current_stream().wait_stream(self._shared_stream)
            out = out + shared_out

        if self.tp_size > 1:
            out = self.allreduce(out)

        return out.view(orig_shape)
