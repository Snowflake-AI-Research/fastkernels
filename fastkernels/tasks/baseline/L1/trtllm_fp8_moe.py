"""FLASHINFER_TRTLLM fp8 block-scale MoE kernel wrapper (Blackwell sm100).

Mirrors vLLM's ``TrtLlmFp8Experts`` — the fp8 MoE backend vLLM's oracle
selects on Blackwell for block-fp8 W8A8 + plain TP (see
``vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`` and
``.../fused_moe/oracle/fp8.py``). On sm100 the oracle leaves
``FLASHINFER_TRTLLM`` first among the CUDA backends; only on Hopper (sm90)
does it force ``TRITON`` to the front. fastkernels' Triton
``VllmFusedExperts`` is the correct Hopper choice but the WRONG (slowest-
priority) backend on Blackwell, and its fp8 accumulation differs from the
TRTLLM-Gen kernel — so on B200 we call the same flashinfer kernel vLLM does.

This wraps ``flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe`` with
BlockMajorK-shuffled weights + DeepSeekFp8 activation scales. The routed
variant consumes ALREADY-SELECTED ``topk_ids``/``topk_weights`` (packed) — it
does not re-route — so fastkernels' ``GroupedTopK`` output feeds it directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_FP8_BLOCK = 128


def trtllm_fp8_moe_available() -> bool:
    """True iff the TRTLLM fp8 block-scale routed-MoE path should be used.

    Matches vLLM's gate (``TrtLlmFp8ExpertsBase._supports_current_device``):
    CUDA + Blackwell sm100 family (device capability major == 10) + flashinfer
    with the trtllm fused-MoE entry points. Hopper (major 9) and every other
    arch keep the Triton ``VllmFusedExperts`` path (vLLM's oracle forces TRITON
    to the front there). Plain TP only — fastkernels does not use expert
    parallelism, so ``local_expert_offset=0`` / ``local_num_experts=E`` hold.
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer.fused_moe as fm  # noqa: F401
        return all(
            hasattr(fm, f)
            for f in ("trtllm_fp8_block_scale_routed_moe", "trtllm_fp8_block_scale_moe")
        )
    except Exception:
        return False


def _swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    """Flip the two row-halves of a ``[E, 2*N, ...]`` tensor: ``[gate; up]`` ->
    ``[up; gate]``. FlashInfer's fused-MoE kernels expect ``w31`` ordering,
    while fastkernels' loader stores ``w13`` (gate at offset 0, up at offset N;
    ``DeepSeekMoE._w13_weight_loader``). Applied to the w13 WEIGHT and its
    block SCALE; w2 is untouched. Mirrors ``swap_w13_to_w31``
    (vllm/.../quantization/utils/flashinfer_utils.py:43-46).
    """
    return (
        x.reshape(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])
        .flip(dims=[1])
        .reshape(x.shape)
    )


def prepare_trtllm_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform raw loaded fp8 expert weights + fp32 block scales into the
    FlashInfer BlockMajorK layout the TRTLLM kernel consumes.

    Inputs (fastkernels layout, per-TP):
      w13       [E, 2N, K]  float8_e4m3fn    (gate;up)
      w2        [E, K, N]   float8_e4m3fn
      w13_scale [E, 2N/128, K/128] float32
      w2_scale  [E, K/128, N/128]  float32
    Returns (gemm1_weights, gemm2_weights, gemm1_weights_scale, gemm2_weights_scale):
      w13_fi    [E, K/128, 2N, 128] float8_e4m3fn   (BlockMajorK)
      w2_fi     [E, N/128, K, 128]  float8_e4m3fn   (BlockMajorK)
      w13_scale [E, 2N/128, K/128]  float32         (w31-swapped, clamped)
      w2_scale  [E, K/128, N/128]   float32         (clamped)

    Mirrors ``_shuffle_deepseek_fp8_moe_weights`` + ``prepare_fp8_moe_layer_for_fi``
    (vllm/.../quantization/utils/flashinfer_utils.py:333-366,444-535). The
    scales stay fp32 (DeepSeekFp8 needs no int32/UE8M0 packing of weight
    scales); only w13's scale is swapped, and both are clamped to guard
    dead-expert near-zero scales (flashinfer_utils.py:530-533).

    IMPORTANT: pass the RAW loaded fp8 weights. Do NOT run the UE8M0 in-place
    requant (``postprocess_fp8_weights_batched``) first — vLLM's TRTLLM path
    uses the un-requantized checkpoint weights, so requanting would diverge.
    """
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import convert_to_block_layout

    E = w13.shape[0]
    two_n, k = w13.shape[1], w13.shape[2]
    n = w2.shape[2]

    # Step 1 — w31 swap for the w13 weight and its scale (w2 untouched).
    w13 = _swap_w13_to_w31(w13.contiguous())
    w13_scale_out = _swap_w13_to_w31(w13_scale.contiguous()).clone().clamp_(min=1e-10)
    w2_scale_out = w2_scale.contiguous().clone().clamp_(min=1e-10)

    # Step 2 — BlockMajorK shuffle of the WEIGHTS ONLY (epilogue_tile_m=64,
    # block_k=128). Build the output 4D tensor and fill per-expert to bound
    # peak memory (avoids torch.stack holding E copies).
    def _shuffle_stack(w: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        out = torch.empty(
            E, cols // _FP8_BLOCK, rows, _FP8_BLOCK,
            dtype=torch.uint8, device=w.device,
        )
        w_u8 = w.view(torch.uint8)
        for i in range(E):
            out[i] = convert_to_block_layout(
                shuffle_matrix_a(w_u8[i].contiguous(), 64), _FP8_BLOCK
            )
        return out.view(torch.float8_e4m3fn)

    w13_fi = _shuffle_stack(w13, two_n, k)   # [E, K/128, 2N, 128]
    w2_fi = _shuffle_stack(w2, k, n)          # [E, N/128, K, 128]
    return w13_fi, w2_fi, w13_scale_out, w2_scale_out


def _pack_topk_ids_weights(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Pack routing into the int32 format the routed kernel expects:
    ``(expert_id << 16) | (bf16(weight) bits & 0xFFFF)``. Reimplements vLLM's
    ``trtllm_moe_pack_topk_ids_weights`` (vllm/.../fused_moe/utils.py:372-427).
    Weights are truncated to bf16 precision inside the pack (the kernel decodes
    bf16 weights for the top-k combine); expert ids <= 255 fit in 16 bits.
    """
    w16 = (
        topk_weights.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
        & 0xFFFF
    )
    return (topk_ids.to(torch.int32) << 16) | w16


class TrtllmFp8MoE(nn.Module):
    """flashinfer TRTLLM-Gen fp8 block-scale routed MoE.

    ``forward`` takes pre-quantized fp8 activations + DeepSeekFp8 scale, the
    BlockMajorK weights from :func:`prepare_trtllm_moe_weights`, and the routed
    (already-selected) top-k ids/weights. It performs GEMM1 + SiLU-gate +
    GEMM2 + top-k weighted-sum in one kernel (``do_finalize=True``) and returns
    the finalized ``[M, hidden]`` output — so the caller must NOT run a
    separate ``moe_sum``. ``routed_scaling_factor`` is applied by the caller
    post-experts (kernel arg left ``None``), matching vLLM.
    """

    def forward(
        self,
        a_fp8: torch.Tensor,          # [M, K] float8_e4m3fn (per-token-group quantized)
        a_scale_t: torch.Tensor,      # [K//128, M] float32 (DeepSeekFp8 layout)
        w13_fi: torch.Tensor,         # [E, K/128, 2N, 128] float8_e4m3fn
        w13_scale: torch.Tensor,      # [E, 2N/128, K/128] float32
        w2_fi: torch.Tensor,          # [E, N/128, K, 128] float8_e4m3fn
        w2_scale: torch.Tensor,       # [E, K/128, N/128] float32
        topk_weights: torch.Tensor,   # [M, top_k] float32
        topk_ids: torch.Tensor,       # [M, top_k] int32
        num_experts: int,
        top_k: int,
        intermediate_size: int,       # N per TP
    ) -> torch.Tensor:
        import flashinfer.fused_moe as fm
        from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout

        M, K = a_fp8.shape
        packed_ids = _pack_topk_ids_weights(topk_ids, topk_weights)
        out = torch.empty(M, K, dtype=torch.bfloat16, device=a_fp8.device)
        fm.trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_ids,
            routing_bias=None,
            hidden_states=a_fp8,
            hidden_states_scale=a_scale_t,
            gemm1_weights=w13_fi,
            gemm1_weights_scale=w13_scale,
            gemm2_weights=w2_fi,
            gemm2_weights_scale=w2_scale,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,   # applied post-experts by the caller
            routing_method_type=1,        # ignored for the routed (pre-selected) variant
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            do_finalize=True,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            output=out,
        )
        return out

    def forward_monolithic(
        self,
        a_fp8: torch.Tensor,          # [M, K] float8_e4m3fn
        a_scale_t: torch.Tensor,      # [K//128, M] float32 (DeepSeekFp8 layout)
        w13_fi: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_fi: torch.Tensor,
        w2_scale: torch.Tensor,
        router_logits: torch.Tensor,  # [M, num_experts] raw gate logits (pre-sigmoid)
        routing_bias: torch.Tensor | None,  # e_score_correction_bias [num_experts]
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        intermediate_size: int,
    ) -> torch.Tensor:
        """Monolithic TRTLLM MoE: routing (sigmoid + noaux_tc grouped top-k +
        norm) is done INSIDE the kernel from raw ``router_logits``, exactly as
        vLLM's DeepSeek/GLM CUDA path (``trtllm_fp8_block_scale_moe`` with
        ``routing_method_type=DeepSeekV3``). This is BIT-IDENTICAL to vLLM,
        whereas the pre-routed variant truncates the top-k combine weights to
        bf16 (``_pack_topk_ids_weights``) — a ~1-ULP error that scales with the
        expert-output magnitude. ``routed_scaling_factor`` is left at 1.0 here
        and applied post-experts by the caller (matching vLLM's runner)."""
        import flashinfer.fused_moe as fm
        from flashinfer.fused_moe import (
            Fp8QuantizationType, WeightLayout, RoutingMethodType,
        )
        res = fm.trtllm_fp8_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=routing_bias,
            hidden_states=a_fp8,
            hidden_states_scale=a_scale_t,
            gemm1_weights=w13_fi,
            gemm1_weights_scale=w13_scale,
            gemm2_weights=w2_fi,
            gemm2_weights_scale=w2_scale,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=1.0,    # kernel no-op; real factor applied post
            routing_method_type=int(RoutingMethodType.DeepSeekV3),
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            do_finalize=True,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        )
        return res[0] if isinstance(res, (list, tuple)) else res
