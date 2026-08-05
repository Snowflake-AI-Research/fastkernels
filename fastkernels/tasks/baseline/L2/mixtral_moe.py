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
        # OFF by default, but note this is a genuine trade rather than a clear
        # loss -- scored against the 0.95x bar it fails FEWER scenarios than the
        # Triton default does. Idle-host A/B at tp=2, Triton -> trtllm:
        #
        #     single-request  (1 tok/step)    0.8588x -> 1.0141x   trtllm passes
        #     fixed-batch-32  (32 tok/step)   0.9123x -> 1.0167x   trtllm passes
        #     long-context    (64 tok/step)   1.0048x -> 0.9711x   both pass
        #     mixed           (100s/step)     1.0128x -> 0.8812x   only Triton
        #
        # So Triton fails 2 (single-request, fixed-batch-32) and trtllm fails 1
        # (mixed). Do not score this by "which side regressed" -- long-context's
        # 1.0048 -> 0.9711 is a regression but still clears the bar, and counting
        # it as a failure is what made this look like a 2-for-2 wash. Flipping the
        # default would close Mixtral's entire latency axis at the cost of mixed
        # throughput; it stays off because that trades a passing throughput
        # scenario for a passing latency one, which is a product call.
        #
        # trtllm-gen's low-latency MoE wins while the block is
        # weight-bandwidth-bound at a few tokens; the Triton grouped GEMM takes
        # over once there is enough work per expert to be compute-bound. Keeping
        # both layouts to dispatch on token count was measured too: it recovered
        # mixed to 0.9987x and single-request to 1.0262x, but it is
        # arithmetically infeasible -- keeping both layouts DOUBLES the model.
        # Mixtral's 47B params are almost entirely experts: 1.41 GiB per layer x
        # 32 layers = ~45 GiB per rank, and the KV sizing inputs confirm it (live
        # allocation 90.0 GiB with the duplicate vs 48.0 GiB without, so available
        # KV fell 108.9 -> 66.6 GiB and token slots 1,784,528 -> 1,091,008). That
        # starved the scheduler and regressed long-context to 0.9316x and
        # fixed-batch-32 to 0.9044x. Do not retry dispatch-by-size for a model
        # whose experts dominate its weights -- and note it cannot be made free
        # by sharing one copy: the bf16 kernel hard-asserts BlockMajorK. Both
        # entry points accept ``weight_layout=MajorK`` with
        # ``use_shuffled_weight=False``, but passing the plain Triton layout fails
        # at every token count with ``Check failed: (weight_layout ==
        # batchedGemm::gemm::MatrixLayout::BlockMajorK) is false``.
        #
        # What is NOT the cause of the mixed deficit, all measured:
        #   * the kernel degrading at prefill widths -- per-token cost is flat
        #     from 1024 to 16384 tokens (0.663 -> 0.607 us/tok, slightly
        #     improving), and chunking the call is strictly worse at every size.
        #   * a different entry point -- vLLM's oracle logs
        #     ``TrtLlmBf16ExpertsMonolithic``, which is the router-side
        #     ``trtllm_bf16_moe`` we already drive. It has no token-count guard
        #     and no Triton fallback, so vLLM runs this same kernel at every
        #     width and still reaches 15,535 tok/s where we reach 13,711.
        #   * a different autotune bucket set -- both engines run
        #     ``max_num_batched_tokens=16384`` at dp=1, so vLLM's
        #     ``fi_moe_largest_bucket`` == the 16384 we pass.
        #   * KV sizing -- single-layout trtllm gets 1,780,736 token slots vs
        #     Triton's 1,784,528, so this is not eviction.
        #
        # That leaves something *around* the call rather than in it. One clue:
        # trtllm-gen logs ``cooperative launch SM allocation: 140 SMs used for
        # MoE, 8 SMs reserved for overlapping kernels`` -- a launch holding 140
        # of 148 SMs cannot be co-scheduled, which would explain why it wins at
        # bs=1 (nothing to overlap) and loses on mixed (other work per step). It
        # cannot be the whole story, since vLLM runs the same cooperative kernel
        # and its mixed is fast; the question is what *we* overlap with the MoE
        # that vLLM does not. Compare a mixed-step kernel table under both flags
        # reading wall-vs-sum-of-kernels, not kernel times -- every finding above
        # came from a decode-only bs=1 profile, and mixed is what regresses.
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
        """Shuffle the experts into trtllm-gen's BlockMajorK layout, in place.

        Mirrors vLLM's ``convert_to_unquantized_kernel_format`` for the
        FLASHINFER_TRTLLM backend. This REPLACES w13/w2 rather than keeping a
        second copy, so the Triton path is unreachable afterwards -- guarded by
        ``use_trtllm``. Keeping both layouts is not an option for this model; see
        __init__ for the measured reason (it doubles a 47B model).
        """
        if not self.use_trtllm or self._trtllm_weights_ready:
            return
        w13_t, w2_t = prepare_trtllm_bf16_moe_weights(self.w13.data, self.w2.data)
        # REPLACE, not duplicate: see __init__ for why keeping both is infeasible
        # here. The Triton path is unreachable afterwards.
        self.w13 = nn.Parameter(w13_t, requires_grad=False)
        self.w2 = nn.Parameter(w2_t, requires_grad=False)
        self._trtllm_weights_ready = True

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core MoE logic, callable from both eager and custom-op paths."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.gate(hidden_states)

        # Which kernel is fixed at load time, not per step: the trtllm-gen path
        # only exists once ``process_weights_after_loading`` has replaced w13/w2
        # with the BlockMajorK layout, and that replacement is irreversible.
        if self._trtllm_weights_ready:
            out = self.trtllm_moe(
                hidden_states, self.w13, self.w2, router_logits,
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
