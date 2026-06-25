"""FP8 linear (block-scaled FP8 matrix multiply) with vLLM-parity dispatch.

Registers the FP8 GEMM as a ``torch.library`` custom op so it stays
**opaque** during ``torch.compile`` tracing (Inductor does not attempt
to inline or fuse it).  At runtime the real DeepGEMM kernel executes —
matching vLLM's approach of using ``torch.ops.vllm.fp8_gemm_nt_op``.

For DeepSeek-V3.2 dense FP8 linears (block-scaled, BF16-in, FP8 weight,
BF16-out), vLLM's ``W8A8BlockFp8LinearOp.apply`` uses a two-tier dispatch:
  * ``M < 32``: FlashInfer's ``fp8_blockscale_gemm_sm90`` (swapAB kernel
    with **internal** activation quantization).  Required for accuracy
    parity with vLLM in low-batch decode and small prefill.
  * ``M >= 32``: DeepGEMM ``fp8_gemm_nt`` with **external** activation
    quantization (``QuantFP8`` -> column-major scales -> ``fp8_gemm_nt``
    with ``disable_ue8m0_cast=not is_deep_gemm_e8m0_used()``).

Mirroring this dispatch is necessary for bit-equivalence with vLLM:
otherwise a single FP8 linear at layer 0 already drifts by ``max|Δ|≈0.25``
on a 16-token batch, which compounds into expert-selection mismatches in
the noaux_tc grouped-topk path of every subsequent MoE layer.
"""

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

# DeepGEMM provides the FP8 fast path on Hopper+; on environments
# without it the BF16 / Triton fallbacks still work.  Import lazily
# so importing this module from a non-FP8 codepath (e.g. the BF16
# Jamba-tiny-dev MoE) doesn't fail on machines without DeepGEMM.
try:
    import deep_gemm
    _HAS_DEEP_GEMM = True
except ImportError:  # pragma: no cover -- exercised at import time
    deep_gemm = None  # type: ignore[assignment]
    _HAS_DEEP_GEMM = False

from .fp8_grouped_gemm_contiguous import _is_deep_gemm_e8m0_used


_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_GROUP_SIZE: tl.constexpr = 128

# Match vLLM's ``per_token_group_quant_fp8`` default:
# ``vllm/.../fp8_utils.py:860`` ``eps: float = 1e-10``.  The previous
# value (1e-12) caused the per-token scale of all-zero rows to differ
# from vLLM by exactly 100x, which propagates as a constant offset into
# the FP8 GEMM output and contributes to layer-0 divergence.
_QUANT_EPS = 1e-10


# ---------------------------------------------------------------------------
# FlashInfer FP8 blockscale GEMM (M < 32 swapAB) - resolved lazily
# ---------------------------------------------------------------------------

_FLASHINFER_RESOLVED = False
_FLASHINFER_FN: object | None = None


def _maybe_get_flashinfer_fp8_gemm():
    """Return ``flashinfer.gemm.fp8_blockscale_gemm_sm90`` if importable +
    enabled by env, otherwise ``None``.

    Mirrors vLLM's enablement gate in
    ``vllm/utils/flashinfer.py:is_flashinfer_fp8_blockscale_gemm_supported``:
    ``VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`` (default 1) AND
    ``has_flashinfer_fp8_blockscale_gemm()`` (Hopper + flashinfer wheel
    exposes ``fp8_blockscale_gemm_sm90``).
    """
    global _FLASHINFER_RESOLVED, _FLASHINFER_FN
    if _FLASHINFER_RESOLVED:
        return _FLASHINFER_FN
    _FLASHINFER_RESOLVED = True

    if os.environ.get("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1") != "1":
        return None
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability()
    if cap[0] != 9:  # Hopper only — same gate as vLLM.
        return None
    try:
        from flashinfer.gemm import fp8_blockscale_gemm_sm90
        _FLASHINFER_FN = fp8_blockscale_gemm_sm90
    except Exception:
        _FLASHINFER_FN = None
    return _FLASHINFER_FN

# ---------------------------------------------------------------------------
# Register FP8 GEMM as torch.library custom ops (opaque to Inductor).
# This mirrors vLLM's direct_register_custom_op for fp8_gemm_nt_op.
# ---------------------------------------------------------------------------

_fp8_lib = torch.library.Library("fastkernels_fp8", "DEF")

_fp8_lib.define(
    "fp8_gemm_nt(Tensor q_input, Tensor input_scale, "
    "Tensor weight, Tensor weight_scale, Tensor! output) -> ()"
)


def _fp8_gemm_nt_impl(q_input, input_scale, weight, weight_scale, output):
    # vLLM passes ``is_deep_gemm_e8m0_used`` explicitly so DeepGEMM's
    # ``disable_ue8m0_cast`` flag matches the SF format used at quantization
    # time (see ``vllm/utils/deep_gemm.py:fp8_gemm_nt`` -> forwards
    # ``disable_ue8m0_cast=not use_ue8m0`` to ``_fp8_gemm_nt_impl``). Without
    # this, DeepGEMM falls back to its module-level default which can re-cast
    # the SFs and silently change the GEMM result by ~1e-3 absolute.
    use_ue8m0 = _is_deep_gemm_e8m0_used()
    deep_gemm.fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        disable_ue8m0_cast=not use_ue8m0,
    )


_fp8_lib.impl("fp8_gemm_nt", _fp8_gemm_nt_impl, "CUDA")


@torch.library.impl(_fp8_lib, "fp8_gemm_nt", "Meta")
def _fp8_gemm_nt_meta(q_input, input_scale, weight, weight_scale, output):
    pass


# ---------------------------------------------------------------------------
# FlashInfer FP8 blockscale GEMM custom op (BF16 in, FP8 weight, BF16 out).
# Used for M < 32 so fastkernels picks the same swapAB kernel as vLLM's
# ``W8A8BlockFp8LinearOp.apply`` (see ``vllm/.../fp8_utils.py:402-407``).
# ---------------------------------------------------------------------------

_fp8_lib.define(
    "flashinfer_blockscale_gemm(Tensor input_bf16, Tensor weight_fp8, "
    "Tensor weight_scale, Tensor! output) -> ()"
)


def _flashinfer_blockscale_gemm_impl(input_bf16, weight_fp8, weight_scale,
                                     output):
    fn = _maybe_get_flashinfer_fp8_gemm()
    assert fn is not None, "FlashInfer FP8 blockscale GEMM not available"
    fn(
        input=input_bf16,
        weight=weight_fp8,
        input_scale=None,  # internal quantization
        weight_scale=weight_scale,
        out=output,
        out_dtype=torch.bfloat16,
    )


_fp8_lib.impl("flashinfer_blockscale_gemm", _flashinfer_blockscale_gemm_impl,
              "CUDA")


@torch.library.impl(_fp8_lib, "flashinfer_blockscale_gemm", "Meta")
def _flashinfer_blockscale_gemm_meta(input_bf16, weight_fp8, weight_scale,
                                     output):
    pass


_fp8_lib.define(
    "per_token_group_quant_fp8(Tensor input, Tensor! output_fp8, "
    "Tensor! output_scale, bool column_major_scales=False) -> ()"
)


def _per_token_group_quant_fp8_op_impl(input, output_fp8, output_scale,
                                       column_major_scales=False):
    _per_token_group_quant_fp8(
        input, output_fp8, output_scale,
        column_major_scales=column_major_scales,
    )


_fp8_lib.impl("per_token_group_quant_fp8", _per_token_group_quant_fp8_op_impl,
              "CUDA")


@torch.library.impl(_fp8_lib, "per_token_group_quant_fp8", "Meta")
def _per_token_group_quant_fp8_op_meta(input, output_fp8, output_scale,
                                       column_major_scales=False):
    pass


@triton.jit
def _fp8_group_quant_kernel(
    x_ptr, out_ptr, scale_ptr,
    stride_x_row, stride_out_row, stride_s_row, stride_s_group,
    num_cols,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_UE8M0: tl.constexpr = True,
):
    pid = tl.program_id(0)
    groups_per_row = num_cols // GROUP_SIZE
    row = pid // groups_per_row
    group = pid % groups_per_row

    x_base = x_ptr + row * stride_x_row + group * GROUP_SIZE
    cols = tl.arange(0, GROUP_SIZE)
    x = tl.load(x_base + cols).to(tl.float32)

    absmax = tl.max(tl.abs(x))
    # Match vLLM's ``per_token_group_quant_fp8`` epsilon (1e-10, see
    # ``vllm/.../fp8_utils.py:860``).  Differs from the previous 1e-12
    # only on all-zero rows but the bias compounds across layers.
    absmax = tl.maximum(absmax, 1e-10)
    scale_raw = absmax / fp8_max
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale_raw))) if USE_UE8M0 else scale_raw

    x_scaled = x / scale
    x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)
    x_fp8 = x_clamped.to(out_ptr.dtype.element_ty)

    out_base = out_ptr + row * stride_out_row + group * GROUP_SIZE
    tl.store(out_base + cols, x_fp8)

    scale_base = scale_ptr + row * stride_s_row + group * stride_s_group
    tl.store(scale_base, scale)


_HAS_VLLM_CUDA_QUANT: bool | None = None


def _check_vllm_cuda_quant() -> bool:
    global _HAS_VLLM_CUDA_QUANT
    if _HAS_VLLM_CUDA_QUANT is None:
        try:
            import vllm._C  # noqa: F401
            _HAS_VLLM_CUDA_QUANT = hasattr(torch.ops, "_C") and hasattr(
                torch.ops._C, "per_token_group_fp8_quant"
            )
        except (ImportError, AttributeError):
            _HAS_VLLM_CUDA_QUANT = False
    return _HAS_VLLM_CUDA_QUANT


def _per_token_group_quant_fp8(x: torch.Tensor,
                               out_fp8: torch.Tensor,
                               out_scale: torch.Tensor,
                               use_ue8m0: bool = True,
                               column_major_scales: bool = False) -> None:
    """In-place per-token-group FP8 quantization.

    When use_ue8m0=True (default), scales are rounded to powers of two
    (UE8M0 format), matching DeepGEMM dense linear expectations.
    When use_ue8m0=False, scales are plain float32 (absmax / fp8_max),
    matching vLLM's Triton MoE activation quantization.

    When column_major_scales=True, the caller has already allocated
    ``out_scale`` with column-major strides (``out_scale.stride(0) == 1``),
    matching vLLM's DeepGEMM dense linear path.  Used for ``Fp8Linear``
    so the SF layout matches DeepGEMM's expectation without a
    post-quant transpose.

    Prefers vLLM's CUDA C++ kernel when available for lower launch overhead;
    falls back to Triton.
    """
    M, K = x.shape
    if _check_vllm_cuda_quant() and x.is_cuda and x.is_contiguous() and use_ue8m0:
        # ``True`` for ``use_ue8m0`` and ``False`` for ``tma_aligned_scales``
        # is what vLLM's ``W8A8BlockFp8LinearOp._run_deepgemm`` ends up
        # passing for ``column_major_scales=True`` without TMA padding (the
        # scale buffer is allocated by ``_alloc_colmajor_scale`` below with
        # plain ``(num_groups, M)`` row-major + permute).  Match vLLM's
        # ``eps=1e-10`` (fastkernels previously used ``1e-12``).
        torch.ops._C.per_token_group_fp8_quant(
            x, out_fp8, out_scale, int(_GROUP_SIZE),
            _QUANT_EPS, _FP8_INFO.min, _FP8_INFO.max, True,
            column_major_scales, False,
        )
        return

    groups_per_row = K // _GROUP_SIZE
    _fp8_group_quant_kernel[(M * groups_per_row,)](
        x, out_fp8, out_scale,
        x.stride(0), out_fp8.stride(0),
        out_scale.stride(0), out_scale.stride(1),
        K,
        fp8_max=_FP8_INFO.max,
        GROUP_SIZE=_GROUP_SIZE,
        USE_UE8M0=use_ue8m0,
    )


class PerTokenGroupQuantFp8(nn.Module):
    """In-place per-token-group FP8 quantization (single Triton/CUDA kernel).

    Public ``nn.Module`` wrapper around the registered
    ``torch.ops.fastkernels_fp8.per_token_group_quant_fp8`` custom op.  L2
    callers that need activation quantization outside of ``Fp8Linear``
    (e.g. ``DeepSeekMoE``, ``SparseAttnIndexer``) should use this module
    instead of importing the private ``_per_token_group_quant_fp8``.
    """

    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        torch.ops.fastkernels_fp8.per_token_group_quant_fp8(
            x.contiguous() if not x.is_contiguous() else x,
            out_fp8, out_scale,
        )


def _alloc_colmajor_scale(M: int, num_groups: int,
                          device: torch.device) -> torch.Tensor:
    """Allocate a per-token-group scale tensor with column-major strides.

    Layout matches vLLM's DeepGEMM dense path
    (``vllm/model_executor/layers/quantization/utils/fp8_utils.py:914-918``):
    physical storage is ``(num_groups, M)`` row-major and we expose it as
    ``(M, num_groups)`` via ``.permute(-1, -2)`` so DeepGEMM's
    ``fp8_gemm_nt`` sees ``stride(0)==1`` SF — i.e. SF columns are
    contiguous, which the kernel expects for TMA loads.
    """
    return torch.empty(
        (num_groups, M), device=device, dtype=torch.float32,
    ).permute(-1, -2)


class _Fp8PrefillBufs:
    """Shared prefill buffers for FP8 activation quantization.

    Since decoder layers execute sequentially, a single set of buffers
    (sized for max_num_batched_tokens) can be reused across all Fp8Linear
    instances, eliminating per-layer dynamic allocation during prefill.
    One instance per unique (K, N) weight shape.

    The scale buffer ``s`` is column-major (matches vLLM's DeepGEMM path).
    """
    __slots__ = ("a", "s", "o")

    def __init__(self, max_tokens: int, K: int, N: int, device: torch.device):
        num_groups = math.ceil(K / 128)
        self.a = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self.s = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self.o = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)


class Fp8Linear(nn.Module):
    """Block-scaled FP8 linear using deep_gemm.fp8_gemm_nt.

    Weights are stored in float8_e4m3fn with pre-processed UE8M0 block scales
    (transformed via deep_gemm.transform_sf_into_required_layout at load time).
    Activations are dynamically quantized to FP8 per-token-group (group=128)
    using in-place ops for CUDA graph compatibility.
    """

    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf: _Fp8PrefillBufs | None = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        """Pre-allocate activation FP8 buffers for CUDA graph capture.

        The scale buffer is **column-major** to match vLLM's DeepGEMM
        dense FP8 path (see ``_alloc_colmajor_scale``).
        """
        num_groups = math.ceil(K / self.BLOCK_SIZE)
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    # Threshold matching vLLM's ``W8A8BlockFp8LinearOp.apply`` /
    # ``_flashinfer_fp8_blockscale_gemm_impl``: below this M the swapAB
    # kernel inside FlashInfer's ``fp8_blockscale_gemm_sm90`` is used; above,
    # external-quant + DeepGEMM ``fp8_gemm_nt``.  The threshold is hard-coded
    # to 32 in vLLM (``fp8_utils.py:308``).
    _FLASHINFER_M_THRESHOLD = 32

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """FP8 block-scaled GEMM with vLLM-parity dispatch.

        * ``M < 32`` and FlashInfer available → FlashInfer swapAB kernel
          (BF16 in, internal quant, FP8 GEMM, BF16 out).  Same path as
          ``vllm/.../fp8_utils.py:_run_flashinfer``.
        * Otherwise → external ``per_token_group_quant_fp8`` (column-major
          UE8M0 scales) + ``deep_gemm.fp8_gemm_nt`` (with
          ``disable_ue8m0_cast`` set per the resolved oracle).  Same path
          as ``vllm/.../fp8_utils.py:_run_deepgemm``.

        FP8 ops are routed through ``torch.ops.fastkernels_fp8.*`` so they stay
        opaque to ``torch.compile`` (mirrors vLLM's
        ``torch.ops.vllm.fp8_gemm_nt_op`` / ``flashinfer_fp8_blockscale_gemm``).
        """
        N, K = weight_fp8.shape
        input_2d = input_bf16.reshape(-1, K)
        M = input_2d.shape[0]
        num_groups = (K + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

        # Mirror vLLM: pick FlashInfer swapAB iff supported AND batch is
        # small AND dims align with what FlashInfer's check requires
        # (``vllm/utils/flashinfer.py:should_use_flashinfer_for_blockscale_fp8_gemm``:
        # weight ``N % 64 == 0`` and ``K % 128 == 0``).
        use_flashinfer = (
            not torch.compiler.is_compiling()
            and input_bf16.dtype == torch.bfloat16
            and weight_fp8.dtype == torch.float8_e4m3fn
            and N % 64 == 0
            and K % 128 == 0
            and M < self._FLASHINFER_M_THRESHOLD
            and _maybe_get_flashinfer_fp8_gemm() is not None
        )

        if use_flashinfer:
            output = torch.empty(
                M, N, dtype=torch.bfloat16, device=input_2d.device,
            )
            torch.ops.fastkernels_fp8.flashinfer_blockscale_gemm(
                input_2d, weight_fp8, weight_scale_inv, output,
            )
            if bias is not None:
                output = output + bias
            return output.view(*input_bf16.shape[:-1], N)

        if not torch.compiler.is_compiling():
            if self._a_buf is not None and M <= self._a_buf.shape[0]:
                q_input = self._a_buf[:M]
                input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
                output = self._o_buf[:M]
            elif self._pf is not None and M <= self._pf.a.shape[0]:
                q_input = self._pf.a[:M]
                input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
                output = self._pf.o[:M]
            else:
                q_input = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input_2d.device)
                input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
                output = torch.empty(M, N, dtype=torch.bfloat16, device=input_2d.device)
        else:
            q_input = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input_2d.device)
            input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
            output = torch.empty(M, N, dtype=torch.bfloat16, device=input_2d.device)

        # ``column_major_scales=True`` matches the SF layout DeepGEMM expects
        # (``vllm/.../fp8_utils.py:289-294`` uses the same setting). The
        # ``input_scale`` buffer was allocated with column-major strides above.
        torch.ops.fastkernels_fp8.per_token_group_quant_fp8(
            input_2d, q_input, input_scale, True,
        )
        torch.ops.fastkernels_fp8.fp8_gemm_nt(
            q_input, input_scale, weight_fp8, weight_scale_inv, output,
        )

        if bias is not None:
            output = output + bias

        return output.view(*input_bf16.shape[:-1], N)


def postprocess_fp8_weights(weight_fp8: torch.Tensor,
                            scale_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize FP8 weights to UE8M0 scale format and transform scale layout
    for DeepGEMM compatibility. Must be called once after weight loading.

    Matches vLLM's requant_weight_ue8m0_inplace + deepgemm_post_process_fp8_weight_block:
    dequantize in float32 for precision, re-quantize with UE8M0 power-of-two scales,
    then transform scale layout.  Handles non-block-aligned shapes via padding.
    """
    N, K = weight_fp8.shape
    block_size = Fp8Linear.BLOCK_SIZE

    scale_rows = math.ceil(N / block_size)
    scale_cols = math.ceil(K / block_size)
    scale = scale_inv[:scale_rows, :scale_cols].to(torch.float32)

    w_padded = weight_fp8
    need_n_pad = (block_size - N % block_size) % block_size
    need_k_pad = (block_size - K % block_size) % block_size
    if need_n_pad or need_k_pad:
        w_padded = torch.nn.functional.pad(
            weight_fp8.view(torch.int8),
            (0, need_k_pad, 0, need_n_pad),
        ).view(torch.float8_e4m3fn)

    if need_n_pad or need_k_pad:
        w_view = w_padded.view(
            math.ceil(N / block_size + need_n_pad / block_size), block_size,
            math.ceil(K / block_size + need_k_pad / block_size), block_size,
        )
    else:
        w_view = w_padded.view(scale_rows, block_size, scale_cols, block_size)

    w_f32 = w_view.to(torch.float32) * scale[:, None, :, None]

    w_f32_flat = w_f32.reshape(-1, w_f32.shape[2] * block_size)
    if need_n_pad or need_k_pad:
        w_f32_flat = w_f32_flat[:N, :K].contiguous()

    # Mirror vLLM: ``use_ue8m0`` and ``disable_ue8m0_cast`` are both keyed off
    # the same oracle (``is_deep_gemm_e8m0_used``), so the requant + the
    # SF layout transform agree. Keeps weights consistent with what the GEMM
    # kernel expects on this build/arch.
    use_ue8m0 = _is_deep_gemm_e8m0_used()
    w_fp8_new, scale_ue8m0 = deep_gemm.per_block_cast_to_fp8(
        w_f32_flat, use_ue8m0=use_ue8m0,
    )

    recipe = (1, block_size, block_size)
    scale_transformed = deep_gemm.transform_sf_into_required_layout(
        sf=scale_ue8m0.unsqueeze(0),
        mn=N,
        k=K,
        recipe=recipe,
        num_groups=1,
        is_sfa=False,
        disable_ue8m0_cast=not use_ue8m0,
    ).squeeze(0)

    return w_fp8_new, scale_transformed


def postprocess_fp8_weights_batched(weight_fp8: torch.Tensor,
                                    scale_inv: torch.Tensor) -> None:
    """Re-quantize 3D MoE weights [E, N, K] to UE8M0 scales in-place,
    then transform scale layout for DeepGEMM. Matches vLLM's
    requant_weight_ue8m0_inplace + deepgemm_post_process_fp8_weight_block."""
    assert weight_fp8.ndim == 3
    E, N, K = weight_fp8.shape
    block_size = Fp8Linear.BLOCK_SIZE

    scale_rows = math.ceil(N / block_size)
    scale_cols = math.ceil(K / block_size)

    use_ue8m0 = _is_deep_gemm_e8m0_used()

    for idx in range(E):
        w_q = weight_fp8[idx]
        s_old = scale_inv[idx, :scale_rows, :scale_cols]

        s_float = s_old.to(torch.float32)
        s_exp = torch.repeat_interleave(s_float, block_size, dim=0)[:N]
        s_exp = torch.repeat_interleave(s_exp, block_size, dim=1)[:, :K]
        w_dq = w_q.to(torch.float32) * s_exp

        w_requant, s_requant = deep_gemm.per_block_cast_to_fp8(
            w_dq, use_ue8m0=use_ue8m0,
        )
        w_q.copy_(w_requant)
        s_old.copy_(s_requant)

    recipe = (1, block_size, block_size)
    scale_transformed = deep_gemm.transform_sf_into_required_layout(
        sf=scale_inv[:, :scale_rows, :scale_cols],
        mn=N,
        k=K,
        recipe=recipe,
        num_groups=E,
        is_sfa=False,
        disable_ue8m0_cast=not use_ue8m0,
    )
    scale_inv[:, :scale_rows, :scale_cols].copy_(scale_transformed)
