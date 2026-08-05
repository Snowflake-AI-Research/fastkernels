"""Mixtral Mixture-of-Experts block with fused Triton grouped GEMM."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.linear import Linear
from ..L1.topk_softmax import TopKSoftmax
from ..L1.trtllm_bf16_moe import (
    ROUTING_RENORMALIZE,
    TrtLlmBf16MoE,
    prepare_trtllm_bf16_moe_weights,
    trtllm_bf16_moe_supported,
)
from ..L2.fused_experts import FusedExperts


class MixtralMoE(nn.Module):
    """Mixture-of-Experts with fused Triton grouped GEMM.

    Weights:
      w13: [E, 2*intermediate_per_tp, hidden_size] -- gate (w1) and up (w3) stacked
      w2:  [E, hidden_size, intermediate_per_tp]
    """

    # Diagnostic only (FASTKERNELS_MOE_DEBUG=1): records which kernel each
    # distinct token count resolved to. Class-level so all 32 layers share one
    # set and each size is reported once. Note the capture loop walks batch
    # sizes largest-first, so filter small if that is what you care about.
    _dbg_seen: set = set()

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.intermediate_size // tp

        self.gate = Linear(config.hidden_size, config.num_local_experts, bias=False)

        self.w13 = nn.Parameter(torch.empty(
            config.num_local_experts, 2 * self.intermediate_per_tp, config.hidden_size,
        ))
        self.w13.weight_loader = self._w13_weight_loader

        self.w2 = nn.Parameter(torch.empty(
            config.num_local_experts, config.hidden_size, self.intermediate_per_tp,
        ))
        self.w2.weight_loader = self._w2_weight_loader

        self.topk_softmax = TopKSoftmax()
        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

        # trtllm-gen BF16 MoE: what vLLM 0.26 selects for Mixtral on Blackwell
        # (its oracle logs "Using FlashInfer TRTLLM Unquantized MoE backend"
        # ahead of the Triton fused_moe path). Our Triton MoE is where Mixtral's
        # decode deficit lives: profiled at tp=2 bs=1, ``_fused_moe_kernel`` was
        # 55.4% of the step (64 calls x 37.8 us = 2.42 ms of 4.37 ms), and the
        # step was 100% GPU-bound -- total self CUDA time equalled wall time, so
        # there is no host or idle slack to reclaim. The MoE kernel IS the gap.
        #
        # OFF by default anyway, because adopting vLLM's choice is not a free
        # win here. Idle-host A/B at tp=2, Triton -> trtllm:
        #
        #     single-request  (1 tok/step)    0.8588x -> 1.0141x   trtllm wins
        #     fixed-batch-32  (32 tok/step)   0.9123x -> 1.0167x   trtllm wins
        #     long-context    (64 tok/step)   1.0048x -> 0.9711x   triton wins
        #     mixed           (100s/step)     1.0128x -> 0.8812x   triton wins
        #
        # trtllm-gen's low-latency MoE wins while the block is
        # weight-bandwidth-bound at a few tokens; the Triton grouped GEMM takes
        # over once there is enough work per expert to be compute-bound. Keeping
        # both layouts to dispatch on token count was measured too
        # (FASTKERNELS_MOE_TRTLLM_MAX_TOKENS): it recovered mixed to 0.9987x and
        # single-request to 1.0262x, but the duplicate experts cost ~1.4 GiB per
        # rank out of the KV cache and that pushed long-context to 0.9316x and
        # fixed-batch-32 to 0.9044x -- so all three configurations fail exactly
        # two of the four scenarios. Closing this properly needs trtllm-gen to be
        # competitive at large token counts (vLLM reaches 15,535 tok/s on mixed
        # with it, against our 13,690), which an isolated microbenchmark of the
        # kernel could not reproduce in either entry point.
        self.use_trtllm = (
            trtllm_bf16_moe_supported()
            and os.environ.get("FASTKERNELS_MIXTRAL_TRTLLM_MOE", "0") == "1"
        )
        self.trtllm_moe = (
            TrtLlmBf16MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                intermediate_size_per_partition=self.intermediate_per_tp,
                routing_method_type=ROUTING_RENORMALIZE,
            )
            if self.use_trtllm
            else None
        )
        self._trtllm_weights_ready = False
        # Crossover between the two MoE kernels, in tokens per step. Measured
        # (see process_weights_after_loading): trtllm-gen wins at 32 tokens and
        # loses at 64, so the boundary sits between them. Env-overridable
        # because the crossover is a property of the expert shapes, and a model
        # with a different intermediate size will move it.
        self._trtllm_max_tokens = int(
            os.environ.get("FASTKERNELS_MOE_TRTLLM_MAX_TOKENS", "32")
        )

        # Custom-op dispatch for torch.compile (set by engine after model init)
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

    def process_weights_after_loading(self) -> None:
        """Build a second copy of the experts in trtllm-gen's BlockMajorK layout.

        Both layouts are kept because neither kernel wins everywhere. Measured
        end-to-end on an idle B200 at tp=2 (idle-host A/B, Triton -> trtllm):

            single-request  (1 tok/step)    0.8588x -> 1.0141x   trtllm wins
            fixed-batch-32  (32 tok/step)   0.9123x -> 1.0167x   trtllm wins
            long-context    (64 tok/step)   1.0048x -> 0.9711x   triton wins
            mixed           (100s/step)     1.0128x -> 0.8812x   triton wins

        So trtllm-gen's low-latency MoE is the better kernel while the block is
        weight-bandwidth-bound at a handful of tokens, and the Triton grouped
        GEMM takes over once there is enough work per expert to be
        compute-bound. Picking either one alone fixes two scenarios and breaks
        the other two; dispatching on token count passes all four.

        The duplicate costs ~1.4 GiB per rank for Mixtral (0.8% of a B200),
        taken out of the KV cache. Do not "simplify" this by dropping a layout
        without re-running all four scenarios.

        Registered as non-persistent buffers so they stay out of state_dict.
        """
        if not self.use_trtllm or self._trtllm_weights_ready:
            return
        w13_t, w2_t = prepare_trtllm_bf16_moe_weights(self.w13.data, self.w2.data)
        self.register_buffer("w13_trtllm", w13_t, persistent=False)
        self.register_buffer("w2_trtllm", w2_t, persistent=False)
        self._trtllm_weights_ready = True

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core MoE logic, callable from both eager and custom-op paths."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)

        # Branch on token count, not on a captured flag: decode graphs are
        # captured per batch size, so each graph bakes the right kernel, and the
        # eager prefill/mixed path re-evaluates per step.
        _use_trt = (self._trtllm_weights_ready
                    and hidden_states.shape[0] <= self._trtllm_max_tokens)
        if os.environ.get("FASTKERNELS_MOE_DEBUG") == "1":
            _seen = MixtralMoE._dbg_seen
            _key = (hidden_states.shape[0], _use_trt)
            if _key not in _seen and hidden_states.shape[0] <= 64:
                _seen.add(_key)
                print(f"  [moe-dbg] tokens={hidden_states.shape[0]} "
                      f"ready={self._trtllm_weights_ready} "
                      f"thr={self._trtllm_max_tokens} -> "
                      f"{'TRTLLM' if _use_trt else 'triton'}", flush=True)
        if _use_trt:
            out = self.trtllm_moe(
                hidden_states, self.w13_trtllm, self.w2_trtllm, router_logits,
            )
        else:
            topk_weights, topk_ids = self.topk_softmax(
                router_logits, self.top_k, renormalize=True,
            )
            topk_weights = topk_weights.to(hidden_states.dtype)

            out = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
            )

        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)

        return out.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op: inside it Inductor
            # cannot see the collective, so ``AllReduceFusedAddRMSNormPass`` has
            # nothing to match at the MoE end of the layer -- half of every
            # layer's collectives. vLLM keeps its MoE reduction in traced Python
            # for the same reason (``moe_runner._maybe_reduce_final_output``).
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)
