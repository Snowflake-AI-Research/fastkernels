"""Grouped top-k routing for DeepSeek-style MoE.

Mirrors vLLM's reference ``grouped_topk`` implementation in
``vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:84-165``
with full parity for:

* ``scoring_func`` ("sigmoid" / "softmax")
* ``e_score_correction_bias`` — when present, biased scores are used for
  expert selection but original (unbiased) scores are used for routing
  weights, and group score is "sum of top-2" within group; when absent,
  group score is "max within group"
* ``renormalize`` — only renormalize when explicitly requested, with no
  epsilon (matches vLLM)
* ``routed_scaling_factor`` — folded into routing weights when != 1.0
* ``sorted`` — uses ``vllm_is_batch_invariant()`` to pick sorted vs unsorted
  top-k (matches vLLM's batch-invariant mode)

Returns FP32 ``topk_weights`` to match vLLM.

Fast path uses ``_C.grouped_topk`` (verbatim port of vLLM's fused noaux_tc
CUDA kernel — see ``tasks/baseline/L1/csrc/grouped_topk_kernels.cu``).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .csrc import _C


def _is_batch_invariant() -> bool:
    """Return True when running in batch-invariant mode (matches vLLM's
    ``vllm_is_batch_invariant`` helper)."""
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


def _fused_grouped_topk_enabled() -> bool:
    """Mirrors vLLM's enablement gate in
    ``grouped_topk_router.py:95-101``: ``VLLM_USE_FUSED_MOE_GROUPED_TOPK``
    env-var on, CUDA available, fused kernel built into ``_C``.  The
    per-call gates (``num_expert_group<=32 and topk<=32 and
    e_score_correction_bias is not None``) are checked at call-time."""
    return (
        os.environ.get("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1") == "1"
        and torch.cuda.is_available()
    )


class GroupedTopK(nn.Module):
    """Functional grouped top-k router.

    Configuration that is fixed per-MoE layer (``scoring_func``,
    ``renormalize``, ``routed_scaling_factor``) is passed at construction
    so the forward signature stays close to vLLM's. ``e_score_correction_bias``
    is passed at call-time because vLLM treats it as an optional tensor
    argument (None for the no-aux-tc path).
    """

    def __init__(
        self,
        scoring_func: str = "sigmoid",
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        force_sorted: bool = False,
    ) -> None:
        super().__init__()
        if scoring_func not in ("sigmoid", "softmax"):
            raise ValueError(f"Unsupported scoring function: {scoring_func}")
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.force_sorted = force_sorted

    def _postprocess_selected(
        self,
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.force_sorted or _is_batch_invariant()):
            return topk_weights, topk_ids

        gather_ids = topk_ids.to(torch.int64)
        if self.scoring_func == "sigmoid":
            selected_weights = gating_output.gather(1, gather_ids).sigmoid()
        else:
            selected_weights = torch.softmax(gating_output, dim=-1).gather(
                1, gather_ids,
            )

        topk_weights = selected_weights
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids

    def forward(
        self,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None,
        num_expert_group: int,
        topk_group: int,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fast path: fastkernels's fused noaux_tc CUDA kernel
        # (``_C.grouped_topk``, verbatim port of vLLM's
        # ``torch.ops._moe_C.grouped_topk``).  Conditions match
        # ``grouped_topk_router.py:95-101``.  Saves ~10 separate Triton/PyTorch
        # ops per MoE layer per token vs. the eager fallback.
        if (
            e_score_correction_bias is not None
            and num_expert_group <= 32
            and topk <= 32
            and _fused_grouped_topk_enabled()
        ):
            if self.scoring_func == "sigmoid":
                # Kernel applies sigmoid internally.
                topk_weights, topk_ids = _C.grouped_topk(
                    gating_output,
                    num_expert_group,
                    topk_group,
                    topk,
                    self.renormalize,
                    self.routed_scaling_factor,
                    e_score_correction_bias,
                    1,  # scoring_func=1 (sigmoid)
                )
                return self._postprocess_selected(
                    gating_output,
                    topk_weights,
                    topk_ids,
                )
            # Softmax: precompute scores (kernel doesn't have softmax).
            scores = torch.softmax(gating_output, dim=-1)
            topk_weights, topk_ids = _C.grouped_topk(
                scores,
                num_expert_group,
                topk_group,
                topk,
                self.renormalize,
                self.routed_scaling_factor,
                e_score_correction_bias,
                0,  # scoring_func=0 (no activation, scores precomputed)
            )
            return self._postprocess_selected(
                gating_output,
                topk_weights,
                topk_ids,
            )

        # Score computation in the *gating output* dtype (vLLM does *not*
        # cast to FP32 first — see grouped_topk_router.py:117-119).
        if self.scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        else:  # sigmoid
            scores = gating_output.sigmoid()

        num_token = scores.size(0)

        if e_score_correction_bias is not None:
            # Biased scores for selection; original scores for routing weights.
            original_scores = scores
            scores = scores + e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores.view(num_token, num_expert_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
        else:
            # No bias: vLLM uses *max* within group (not sum of top-2).
            group_scores = (
                scores.view(num_token, num_expert_group, -1)
                .max(dim=-1)
                .values
            )

        use_sorted = self.force_sorted or _is_batch_invariant()
        group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, sorted=use_sorted,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
            .reshape(num_token, -1)
        )
        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

        if e_score_correction_bias is not None:
            topk_ids = torch.topk(
                tmp_scores, k=topk, dim=-1, sorted=use_sorted,
            )[1]
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(
                tmp_scores, k=topk, dim=-1, sorted=use_sorted,
            )

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
