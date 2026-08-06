"""Mixtral Mixture-of-Experts block with fused Triton grouped GEMM."""

from __future__ import annotations

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

        # trtllm-gen BF16 MoE is the default, matching vLLM 0.26's kernel choice
        # exactly: its oracle logs "Using FlashInfer TRTLLM Unquantized MoE
        # backend" and selects ``TrtLlmBf16ExpertsMonolithic``, which is the
        # router-side ``trtllm_bf16_moe`` we drive. A/B against the Triton grouped
        # GEMM is still available through ``FASTKERNELS_TRTLLM_BF16_MOE=0``, which
        # ``trtllm_bf16_moe_supported`` honours.
        #
        # Flipping to trtllm closed Mixtral's latency axis (single-request
        # 0.86x -> 1.01x, fixed-batch-32 0.91x -> 1.02x) but at first regressed
        # ``mixed`` throughput to ~0.90x. The cause was NOT the kernel: a clean
        # sequential profile at 1000 sequences showed the trtllm MoE GEMMs at or
        # below the Triton ``_fused_moe_kernel`` in device time. It was CPU
        # dispatch. The fused trtllm call costs 412-676 us of host time per
        # invocation (autotuner tactic lookup + routing config + cooperative
        # launch setup), 2.3-3.7x the Triton path's ~182 us. At 32 layers an
        # EAGER step therefore burns ~21 ms of host dispatch that a CUDA-graph
        # replay would pay only once. vLLM hides it by capturing mixed/decode
        # steps in graphs; our engine graphs decode but capped capture at bs=512,
        # so at high concurrency (mixed reaches ~1000 concurrent seqs) every
        # decode step with bs>512 ran eager and re-paid the dispatch. Raising the
        # capture cap to 1024 for MoE models (see ``capture_cudagraph`` in
        # engine.py) cut the profiled generate 14.4 s -> 12.1 s and lifted
        # GPU-busy 77.7% -> 92.9% with the kernel sum unchanged -- the fix lives
        # there, not here.
        #
        # Do NOT try to recover the small-token regime by keeping both weight
        # layouts and dispatching on token count. It was measured (mixed 0.9987x,
        # single-request 1.0262x) but is arithmetically infeasible: Mixtral's 47B
        # params are almost entirely experts (~45 GiB per rank at tp=2), so a
        # second layout roughly DOUBLES the model -- live allocation 90.0 GiB vs
        # 48.0 GiB, available KV 108.9 -> 66.6 GiB, token slots 1,784,528 ->
        # 1,091,008 -- which starved the scheduler and regressed long-context to
        # 0.9316x and fixed-batch-32 to 0.9044x. It cannot be made free by sharing
        # one copy either: the bf16 kernel hard-asserts BlockMajorK (passing the
        # plain Triton ``weight_layout=MajorK`` fails with ``Check failed:
        # (weight_layout == ...::BlockMajorK) is false``).
        self.use_trtllm = trtllm_bf16_moe_supported()
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
