"""FLASHINFER_TRTLLM NVFP4 block-scale MoE kernel wrapper (Blackwell sm100).

Mirrors vLLM's ``TrtLlmNvFp4ExpertsMonolithic`` — the NVFP4 MoE backend vLLM's
oracle selects on Blackwell for a ``modelopt``/``NVFP4`` checkpoint with plain
TP (see ``vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py``
and ``.../fused_moe/oracle/nvfp4.py``: ``FLASHINFER_TRTLLM`` is first in
``AVAILABLE_BACKENDS`` and ``TrtLlmNvFp4ExpertsMonolithic`` is preferred over
the modular variant whenever all2all / EPLB are off, which is fastkernels'
configuration). Verified on nvidia/GLM-5.2-NVFP4:

    Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend out of potential backends:
    ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', ...]

Monolithic means routing (sigmoid + noaux_tc grouped top-k + renormalize) runs
*inside* the kernel from raw ``router_logits``; the pre-routed
(``..._routed_moe``) variant would truncate the top-k combine weights to bf16,
which is a ~1-ULP error that scales with expert-output magnitude.

Three pieces live here, matching the three vLLM sites:

* :func:`prepare_trtllm_fp4_moe_weights` — ``prepare_nvfp4_moe_layer_for_fi_or_cutlass``
  (reorder ``[w1,w3] -> [w3,w1]``, global-max the activation scales, permute /
  interleave the weights and block scales into TRTLLM-gen layout) plus
  ``TrtLlmNvFp4ExpertsBase.process_weights_after_loading`` (fold the activation
  global scales into the per-expert GEMM alphas).
* :class:`NvFp4Quantize` — the activation quantizer vLLM's
  ``MoEPrepareAndFinalizeNoDPEPMonolithic.prepare`` runs
  (``ops.scaled_fp4_quant`` with ``is_sf_swizzled_layout=False``, because the
  TRTLLM kernel does not accept swizzled input scales).
* :class:`TrtllmFp4MoE` — the ``trtllm_fp4_block_scale_moe`` call itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ``epilogue_tile_m`` for the TRTLLM-gen NVFP4 MoE kernels. Matches
# ``prepare_static_weights_for_trtllm_fp4_moe``
# (vllm/.../quantization/utils/flashinfer_fp4_moe.py:180).
_EPILOGUE_TILE_M = 128
# NVFP4 block size (elements sharing one fp8 scale).
_NVFP4_GROUP = 16

# Permute-index cache keyed by tensor shape, shared across layers. vLLM builds a
# fresh dict per ``prepare_static_weights_for_trtllm_fp4_moe`` call (i.e. per
# layer); the indices depend only on the shape, so hoisting the cache to module
# scope is a pure load-time saving with identical results.
_CACHE_PERMUTE_INDICES: dict[torch.Size, torch.Tensor] = {}


import flashinfer
import flashinfer.fused_moe as _fm


def trtllm_fp4_moe_available() -> bool:
    """True iff the TRTLLM NVFP4 block-scale MoE path should be used.

    Matches vLLM's gate (``TrtLlmNvFp4ExpertsBase._supports_current_device``):
    CUDA + Blackwell sm100 family (device capability major == 10) + flashinfer
    with the trtllm fused-MoE entry points. There is no non-Blackwell fallback
    here: NVFP4 W4A4 has no fastkernels reference kernel, so an unsupported
    device must fail loudly at model build rather than silently diverge.
    """
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    return hasattr(_fm, "trtllm_fp4_block_scale_moe") and hasattr(
        flashinfer, "nvfp4_block_scale_interleave"
    )


def _reorder_w1w3_to_w3w1(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-order concatenated ``[w1, w3]`` rows to ``[w3, w1]``.

    FlashInfer's NVFP4 MoE kernels expect gate/up in ``w3, w1`` order while the
    loader stores ``w1, w3`` (gate at row 0, up at row N). Mirrors
    ``reorder_w1w3_to_w3w1``
    (vllm/.../quantization/utils/flashinfer_fp4_moe.py:31-63) — vLLM swaps in
    place in 64 MiB chunks to bound transient memory; the flip-reshape below is
    the same permutation with one transient copy of each tensor, which is
    ~0.4 GiB for GLM-5.2 at tp=8 and freed immediately.
    """
    out = []
    for t in (weight, scale):
        e, rows = t.shape[0], t.shape[1]
        assert rows % 2 == 0, f"expected even gate/up rows, got {rows}"
        out.append(
            t.reshape(e, 2, rows // 2, *t.shape[2:])
            .flip(dims=[1])
            .reshape(t.shape)
            .contiguous()
        )
    return out[0], out[1]


def prepare_trtllm_fp4_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Transform loaded NVFP4 expert weights into TRTLLM-gen kernel layout.

    Inputs (fastkernels layout, per-TP shard, exactly the on-disk modelopt
    tensors merged into the ``w13`` / ``w2`` pair):

      w13         [E, 2N, K//2]      uint8   (packed fp4, ``[gate; up]``)
      w13_scale   [E, 2N, K//16]     fp8e4m3 (per-16-element block scales)
      w13_scale_2 [E, 2]             fp32    (per-expert global weight scale)
      a13_scale   [E, 2]             fp32    (per-expert static input scale)
      w2          [E, K, N//2]       uint8
      w2_scale    [E, K, N//16]      fp8e4m3
      w2_scale_2  [E]                fp32
      a2_scale    [E]                fp32

    Returns the kernel arguments, keyed by the name
    :class:`TrtllmFp4MoE.forward_monolithic` takes:

      w13, w13_scale, w2, w2_scale — permuted / interleaved
      g1_alphas, g2_alphas         [E] fp32 — GEMM output scales
      g1_scale_c                   [E] fp32 — GEMM1 -> GEMM2 requant scale
      a1_gscale, a2_gscale         [E] fp32 — activation quant global scales

    This fuses three vLLM steps that all run at load time:

    1. ``prepare_nvfp4_moe_layer_for_fi_or_cutlass`` (flashinfer_fp4_moe.py:288):
       ``[w1,w3] -> [w3,w1]``; then, because FLASHINFER_TRTLLM supports a global
       scale factor, ``a13_scale``/``a2_scale`` collapse to the **max over all
       experts** broadcast back to ``[E]``; then the hidden / intermediate
       alignment pads (no-ops for GLM-5.2: hidden 6144 is 256-aligned and the
       tp=8 intermediate 256 is 64-aligned) and finally
       ``prepare_static_weights_for_trtllm_fp4_moe``.
    2. ``make_nvfp4_moe_quant_config`` (oracle/nvfp4.py:504-522): ``a1_gscale =
       1/a13_scale``, ``a2_gscale = 1/a2_scale``, and ``g1/g2_alphas`` alias the
       ``weight_scale_2`` parameters.
    3. ``TrtLlmNvFp4ExpertsBase.process_weights_after_loading``
       (trtllm_nvfp4_moe.py:119-135): ``weight_scale_2 *= input_scale``
       **in place**, which is why the alphas end up carrying both global scales,
       and ``g1_scale_c = g1_alphas * a2_gscale``.

    The reciprocals are taken **before** the in-place multiply in vLLM (step 2
    runs before step 3 and ``1.0 / x`` allocates), so ``a1_gscale`` is
    ``1/a13_scale`` of the *unfolded* input scale — reproduced here by ordering
    the statements the same way.
    """
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    num_experts = w13.shape[0]
    gemm1_rows = w13.shape[1]           # 2N (gated)
    hidden_size = w2.shape[1]           # K
    intermediate = w2.shape[2] * 2      # N (w2's packed last dim x2)
    assert gemm1_rows == 2 * intermediate, (
        f"w13 rows {gemm1_rows} != 2 * intermediate {intermediate}"
    )
    assert hidden_size % 256 == 0, (
        f"hidden_size {hidden_size} is not 256-aligned; vLLM would pad it "
        "(align_trtllm_fp4_moe_hidden_dim_for_fi) and slice activations at "
        "runtime -- not implemented here because no supported config needs it"
    )
    assert intermediate % 64 == 0, (
        f"intermediate_size_per_partition {intermediate} is not 64-aligned; "
        "vLLM would pad it (align_fp4_moe_weights_for_fi) -- not implemented "
        "here because no supported config needs it"
    )

    # --- 1a. [w1, w3] -> [w3, w1] ------------------------------------------
    w13, w13_scale = _reorder_w1w3_to_w3w1(w13.contiguous(), w13_scale.contiguous())

    # --- 1b. Global (all-expert max) activation scales ----------------------
    # ``is_global_sf_supported_for_nvfp4_backend(FLASHINFER_TRTLLM)`` is True,
    # so vLLM reduces over BOTH dims (``.max()``, not ``.max(dim=1)``) and
    # expands back to per-expert. With plain TP every rank holds all experts, so
    # this max is over the same values vLLM sees.
    a13_scale = a13_scale.max().to(torch.float32).expand(num_experts).contiguous()
    a2_scale = a2_scale.max().to(torch.float32).expand(num_experts).contiguous()

    # --- 1c. TRTLLM-gen weight / block-scale permutation -------------------
    w13_fp4 = w13.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_rows, hidden_size // 2
    )
    w13_sf = w13_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_rows, hidden_size // _NVFP4_GROUP
    )
    w2_fp4 = w2.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate // 2
    )
    w2_sf = w2_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate // _NVFP4_GROUP
    )

    w13_out = torch.empty_like(w13_fp4, dtype=torch.uint8)
    w13_sf_out = torch.empty_like(w13_sf, dtype=torch.uint8)
    w2_out = torch.empty_like(w2_fp4, dtype=torch.uint8)
    w2_sf_out = torch.empty_like(w2_sf, dtype=torch.uint8)

    for i in range(num_experts):
        idx = _maybe_get_cached_w3_w1_permute_indices(
            _CACHE_PERMUTE_INDICES,
            w13_fp4[i].view(torch.uint8),
            _EPILOGUE_TILE_M,
            is_gated_act_gemm=True,
        )
        w13_out[i] = w13_fp4[i].view(torch.uint8)[idx.to(w13_fp4.device)]

        sf_idx = _maybe_get_cached_w3_w1_permute_indices(
            _CACHE_PERMUTE_INDICES,
            w13_sf[i].view(torch.uint8),
            _EPILOGUE_TILE_M,
            num_elts_per_sf=_NVFP4_GROUP,
            is_gated_act_gemm=True,
        )
        w13_sf_out[i] = nvfp4_block_scale_interleave(
            w13_sf[i].view(torch.uint8)[sf_idx.to(w13_sf.device)].contiguous()
        ).reshape(w13_sf_out[i].shape)

        idx = get_w2_permute_indices_with_cache(
            _CACHE_PERMUTE_INDICES, w2_fp4[i].view(torch.uint8), _EPILOGUE_TILE_M,
        )
        w2_out[i] = w2_fp4[i].view(torch.uint8)[idx.to(w2_fp4.device)]

        sf_idx = get_w2_permute_indices_with_cache(
            _CACHE_PERMUTE_INDICES,
            w2_sf[i].view(torch.uint8),
            _EPILOGUE_TILE_M,
            num_elts_per_sf=_NVFP4_GROUP,
        )
        w2_sf_out[i] = nvfp4_block_scale_interleave(
            w2_sf[i].view(torch.uint8)[sf_idx.to(w2_sf.device)].contiguous()
        ).reshape(w2_sf_out[i].shape)

    # --- 2. Activation global scales (reciprocals, taken BEFORE the fold) ---
    a1_gscale = (1.0 / a13_scale).contiguous()
    a2_gscale = (1.0 / a2_scale).contiguous()

    # --- 3. Fold the activation scales into the per-expert GEMM alphas ------
    # vLLM warns (and proceeds) when w1 and w3 disagree on the global weight
    # scale, then uses column 0 unconditionally.
    if not torch.allclose(w13_scale_2[:, 0], w13_scale_2[:, 1]):
        import warnings

        warnings.warn(
            "w1_weight_scale_2 must match w3_weight_scale_2. "
            "Accuracy may be affected.",
            stacklevel=2,
        )
    g1_alphas = (w13_scale_2[:, 0].contiguous().to(torch.float32) * a13_scale)
    g2_alphas = (w2_scale_2.contiguous().to(torch.float32) * a2_scale)
    g1_scale_c = (g1_alphas * a2_gscale).contiguous()

    # WEIGHTS stay uint8 (two fp4 values per byte -- the kernel asserts
    # "weights must be fp4 packed in uint8"); only the block SCALES are viewed
    # as fp8_e4m3. vLLM's ``prepare_static_weights_for_trtllm_fp4_moe`` does
    # exactly this: it stacks the permuted ``.view(torch.uint8)`` weights and
    # only re-views the interleaved scales.
    return {
        "w13": w13_out,
        "w13_scale": w13_sf_out.view(torch.float8_e4m3fn),
        "w2": w2_out,
        "w2_scale": w2_sf_out.view(torch.float8_e4m3fn),
        "g1_alphas": g1_alphas.contiguous(),
        "g2_alphas": g2_alphas.contiguous(),
        "g1_scale_c": g1_scale_c,
        "a1_gscale": a1_gscale,
        "a2_gscale": a2_gscale,
    }


class NvFp4Quantize(nn.Module):
    """NVFP4-quantize activations with a static per-tensor global scale.

    Wraps ``torch.ops._C.scaled_fp4_quant`` exactly as vLLM's
    ``_nvfp4_quantize`` -> ``ops.scaled_fp4_quant`` does
    (vllm/_custom_ops.py:scaled_fp4_quant, reached from
    ``MoEPrepareAndFinalizeNoDPEPMonolithic.prepare``).

    ``is_sf_swizzled_layout=False``: ``make_nvfp4_moe_quant_config`` sets
    ``is_scale_swizzled=False`` for FLASHINFER_TRTLLM because "TRTLLM kernel
    does not accept swizzled input quant scales"
    (oracle/nvfp4.py:511-520). vLLM's ``backend`` argument defaults to
    ``"none"``, so the ``m <= 32`` 8x4-scale-layout fast path is NOT taken.

    Returns ``(packed_fp4 [M, K//2] uint8, block_scale [M, K//16] fp8e4m3)``.
    """

    def forward(
        self, x: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm._custom_ops import create_fp4_output_tensors

        assert x.dim() == 2, f"expected 2D activations, got {tuple(x.shape)}"
        m, n = x.shape
        assert n % _NVFP4_GROUP == 0, f"last dim {n} is not a multiple of 16"
        out, out_scale = create_fp4_output_tensors(
            m, n, x.device, False,  # is_sf_swizzled_layout=False
        )
        torch.ops._C.scaled_fp4_quant.out(
            x, global_scale, False, output=out, output_scale=out_scale,
        )
        return out, out_scale.view(torch.float8_e4m3fn)


class TrtllmFp4MoE(nn.Module):
    """flashinfer TRTLLM-gen NVFP4 block-scale MoE (router + experts fused).

    ``forward_monolithic`` performs routing, GEMM1 + SwiGLU, GEMM2, and the
    top-k weighted combine in one kernel (``do_finalize=True``), returning the
    finalized ``[M, hidden]`` bf16 output — so the caller must NOT run a
    separate ``moe_sum``. ``routed_scaling_factor`` is left ``None`` and applied
    post-experts by the caller, matching ``ModelOptNvFp4FusedMoE.apply_monolithic``
    (which forwards ``layer.routed_scaling_factor``; DeepSeek/GLM set it to 1.0
    on the layer and scale afterwards in ``deepseek_v2.py``).
    """

    def __init__(self, tune_max_num_tokens: int = 8192):
        super().__init__()
        # ``fi_moe_largest_bucket`` = max(max_num_tokens * dp_size, 8192); with
        # dp_size == 1 (plain TP) that is max(max_num_batched_tokens, 8192).
        self.tune_max_num_tokens = tune_max_num_tokens

    def forward_monolithic(
        self,
        a_fp4: torch.Tensor,          # [M, K//2] uint8 (packed nvfp4)
        a_scale: torch.Tensor,        # [M, K//16] fp8e4m3 (linear, unswizzled)
        w13: torch.Tensor,            # [E, 2N, K//2] uint8 (packed nvfp4)
        w13_scale: torch.Tensor,      # [E, 2N, K//16] fp8e4m3
        w2: torch.Tensor,             # [E, K, N//2] uint8 (packed nvfp4)
        w2_scale: torch.Tensor,       # [E, K, N//16] fp8e4m3
        g1_alphas: torch.Tensor,      # [E] fp32
        g2_alphas: torch.Tensor,      # [E] fp32
        g1_scale_c: torch.Tensor,     # [E] fp32
        router_logits: torch.Tensor,  # [M, E] raw gate logits (pre-sigmoid)
        routing_bias: torch.Tensor | None,   # e_score_correction_bias [E]
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        intermediate_size: int,       # N per TP
        norm_topk_prob: bool = True,
    ) -> torch.Tensor:
        import flashinfer.fused_moe as fm
        from flashinfer.fused_moe import RoutingMethodType
        from flashinfer.fused_moe.core import ActivationType

        res = fm.trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=routing_bias,
            hidden_states=a_fp4,
            hidden_states_scale=a_scale.view(torch.float8_e4m3fn).reshape(
                *a_fp4.shape[:-1], -1
            ),
            gemm1_weights=w13,
            gemm1_weights_scale=w13_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=g1_scale_c,
            output1_scale_gate_scalar=g1_alphas,
            output2_scale_scalar=g2_alphas,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,   # applied post-experts by the caller
            routing_method_type=int(RoutingMethodType.DeepSeekV3),
            do_finalize=True,
            # ``ActivationType.Swiglu`` — what vLLM's
            # ``activation_to_flashinfer_int`` maps ``MoEActivation.SILU`` to
            # (gated SiLU), i.e. GLM-5.2 / DeepSeek's activation.
            activation_type=int(ActivationType.Swiglu),
            per_token_scale=None,         # static (not per-token) global scale
            tune_max_num_tokens=self.tune_max_num_tokens,
            norm_topk_prob=norm_topk_prob,
        )
        return res[0] if isinstance(res, (list, tuple)) else res
