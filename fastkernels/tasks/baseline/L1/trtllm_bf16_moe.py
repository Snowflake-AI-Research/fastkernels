"""TRTLLM-gen BF16 fused MoE (via FlashInfer, Blackwell only).

This is the kernel vLLM 0.26 actually runs for the unquantized BF16 MoE of
Qwen3-Next and Kimi-Linear on SM100. Its oracle picks the
``FLASHINFER_TRTLLM`` unquantized backend and ``TrtLlmBf16ExpertsMonolithic``,
then autotunes ``flashinfer::trtllm_bf16_moe`` -- visible in a reference run's
log as ``[AutoTuner]: Tuning flashinfer::trtllm_bf16_moe``.

The Triton ``_fused_moe_kernel`` path in :mod:`fused_experts` computes the same
math and is much slower here: it took 22.5% of a Qwen3-Next 32768-token prefill
profile, with a further ~7% in PyTorch's ``mbtopk`` where the reference has the
routing fused into this same kernel.

Beyond the kernel swap, two things differ from the Triton path:

* **Routing is fused in.** ``trtllm_bf16_moe`` takes the raw router logits plus
  the routing description (method, groups, bias, scaling) and does gating, top-k
  and the weighted reduction itself, so the separate gate/top-k/scale steps go
  away. ``routed_scaling_factor`` is applied inside the kernel.
* **A shuffled 4D BlockMajorK weight layout** for the transposed MMA epilogue,
  plus a gate/up row rotation because trtllm-gen defines SwiGLU with the two
  halves in the opposite order. :func:`prepare_trtllm_bf16_moe_weights` is a
  port of vLLM's ``convert_moe_weights_to_flashinfer_trtllm_block_layout``.

Mirrors ``TrtLlmBf16ExpertsMonolithic.apply``
(``vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py``).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

try:
    from flashinfer.fused_moe import trtllm_bf16_moe as _trtllm_bf16_moe

    _TRTLLM_BF16_MOE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _trtllm_bf16_moe = None
    _TRTLLM_BF16_MOE_AVAILABLE = False

# ``RoutingMethodType`` values (vllm/model_executor/layers/fused_moe/config.py).
# The kernel implements each scoring/normalization scheme internally, so the
# caller only names the one its config asks for.
ROUTING_RENORMALIZE = 1
ROUTING_DEEPSEEK_V3 = 2
ROUTING_RENORMALIZE_NAIVE = 4

# ``ActivationType.Swiglu`` == 3 -- what
# ``activation_to_flashinfer_int(MoEActivation.SILU)`` resolves to (vLLM maps SILU
# to its *gated* form). The value matters structurally, not just numerically: the
# launcher derives ``intermediate_size_factor`` from it, so a non-gated id makes
# ``check_weights_shape`` reject w13's 2*I rows.
ACTIVATION_SWIGLU = 3

# vLLM's ``fi_moe_largest_bucket``: ``max(max_num_tokens * dp_size, 8192)``. Both
# engines run ``max_num_batched_tokens=16384`` at dp=1, so vLLM tunes to 16384 and
# we were tuning to the 8192 floor -- a different bucket set, and so potentially a
# different tactic selected for *every* shape including the batch-1 decode ones.
# TODO: plumb ``max_num_batched_tokens`` through from the engine instead of
# restating its value here; L1 cannot import the engine without a cycle.
DEFAULT_TUNE_MAX_NUM_TOKENS = 16384

# ``epilogue_tile_m`` / ``block_k`` from
# ``convert_moe_weights_to_flashinfer_trtllm_block_layout``.
_EPILOGUE_TILE_M = 128
_BLOCK_K = 128


def trtllm_bf16_moe_supported() -> bool:
    """True when the trtllm-gen BF16 MoE kernel can run on this device.

    vLLM gates ``TrtLlmBf16ExpertsBase`` on ``is_device_capability_family(100)``
    plus ``has_flashinfer_trtllm_fused_moe()``, i.e. Blackwell only.
    ``FASTKERNELS_TRTLLM_BF16_MOE=0`` forces the Triton ``fused_experts`` path
    instead, for A/B against the reference.
    """
    if os.environ.get("FASTKERNELS_TRTLLM_BF16_MOE", "1") == "0":
        return False
    if not _TRTLLM_BF16_MOE_AVAILABLE or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def _copy_permuted_expert_to_block_layout(
    out: torch.Tensor,
    expert_uint8: torch.Tensor,
    source_indices: torch.Tensor,
) -> None:
    expert_blocks = expert_uint8.view(
        expert_uint8.shape[0], out.shape[0], _BLOCK_K,
    ).permute(1, 0, 2)
    torch.index_select(
        expert_blocks,
        1,
        source_indices.to(expert_uint8.device),
        out=out,
    )


def prepare_trtllm_bf16_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    is_gated_act_gemm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle BF16 expert weights into FlashInfer's 4D BlockMajorK layout.

    ``w13`` is ``[E, 2*I, H]`` and ``w2`` is ``[E, H, I]`` (the layout the
    checkpoint loaders already produce). Returns
    ``[E, H // 128, 2*I, 128]`` and ``[E, I // 128, H, 128]``.

    Port of vLLM's ``convert_moe_weights_to_flashinfer_trtllm_block_layout``.
    """
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("trtllm-gen BF16 MoE requires bfloat16 weights")

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    cache: dict[torch.Size, torch.Tensor] = {}
    num_experts = w13.shape[0]
    w13_rows, w13_cols = w13[0].view(torch.uint8).shape
    w2_rows, w2_cols = w2[0].view(torch.uint8).shape

    w13_shuffled = torch.empty(
        (num_experts, w13_cols // _BLOCK_K, w13_rows, _BLOCK_K),
        dtype=torch.uint8,
        device=w13.device,
    )
    w2_shuffled = torch.empty(
        (num_experts, w2_cols // _BLOCK_K, w2_rows, _BLOCK_K),
        dtype=torch.uint8,
        device=w2.device,
    )

    for i in range(num_experts):
        w13_expert = w13[i].view(torch.uint8)
        permute = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_expert, _EPILOGUE_TILE_M,
            is_gated_act_gemm=is_gated_act_gemm,
        )
        if is_gated_act_gemm:
            # trtllm-gen's SwiGLU expects [w3; w1] where the checkpoint gives
            # [w1; w3], so rotate the row permutation by half.
            rows = w13_expert.shape[0]
            permute = (permute + rows // 2) % rows
        _copy_permuted_expert_to_block_layout(w13_shuffled[i], w13_expert, permute)

        w2_expert = w2[i].view(torch.uint8)
        _copy_permuted_expert_to_block_layout(
            w2_shuffled[i],
            w2_expert,
            get_w2_permute_indices_with_cache(cache, w2_expert, _EPILOGUE_TILE_M),
        )

    return w13_shuffled.view(torch.bfloat16), w2_shuffled.view(torch.bfloat16)


class TrtLlmBf16MoE(nn.Module):
    """Monolithic trtllm-gen BF16 MoE: routing, both GEMMs and the reduction.

    ``w13``/``w2`` must already be in the shuffled BlockMajorK layout produced
    by :func:`prepare_trtllm_bf16_moe_weights`.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size_per_partition: int,
        routing_method_type: int = ROUTING_RENORMALIZE,
        local_expert_offset: int = 0,
        local_num_experts: int | None = None,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        routed_scaling_factor: float | None = None,
        tune_max_num_tokens: int = DEFAULT_TUNE_MAX_NUM_TOKENS,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.routing_method_type = routing_method_type
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = (
            num_experts if local_num_experts is None else local_num_experts
        )
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.tune_max_num_tokens = tune_max_num_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = _trtllm_bf16_moe(
            routing_logits=router_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            gemm1_weights=w13,
            gemm2_weights=w2,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            activation_type=ACTIVATION_SWIGLU,
            tune_max_num_tokens=self.tune_max_num_tokens,
        )
        return out[0] if isinstance(out, (list, tuple)) else out
