"""W1.58A8 GEMM: int8 activations × packed-2-bit ternary weights.

This is the L1 primitive that powers ``BitLinear``.  It mirrors Microsoft's
reference ``bitnet_int8xint2_linear`` kernel
(``vllm_repo/BitNet/gpu/bitnet_kernels/bitnet_kernels.cu``) and HF's
``transformers.integrations.bitnet.AutoBitLinear`` forward:

    output_bf16 = (act_int8 @ weight_ternary.T) * weight_scale / act_scale

* ``act_int8``    - per-token int8-quantized activations, shape ``(M, K)``.
* ``act_scale``   - bf16 per-row scale ``s = 127 / max|x|``, shape ``(M,)``.
* ``weight``      - uint8 packed-2bit weights, shape ``(N, K/4)``.  Each byte
                    packs 4 consecutive ternary values for a single output
                    row, biased by +1 (so 0/1/2 -> -1/0/+1).
* ``weight_scale``- bf16 per-output-row dequantization scale, shape ``(N,)``.
                    For non-merged ``BitLinear`` this is the HF scalar broadcast
                    over all rows; for fused ``BitLinearMerged`` it carries
                    the per-shard scale broadcast over each shard's rows.

The ``(N, K/4)`` weight layout is intentionally not the HF on-disk layout
(which is ``(N/4, K)`` packed along OUT).  We re-pack at load time so that
every output row's packed bytes are contiguous along K, making the kernel
trivial: each Triton block reads ``BLOCK_K/4`` bytes per row, expands them
to ``BLOCK_K`` int8 ternary values via bit shifts, and does a standard
``tl.dot`` int8x int8 -> int32 accumulation.

This single kernel handles both ``BitLinear`` (single weight scale broadcast)
and ``BitLinearMerged`` (Q/K/V or gate/up fused) with no kernel changes -
the per-shard scaling is encoded by the ``(N,)`` ``weight_scale`` array.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _bitnet_int8xint2_gemm_kernel(
    A_ptr,            # int8,  (M, K)
    A_scale_ptr,      # bf16,  (M,)
    Wp_ptr,           # uint8, (N, K/4)  -- KN-packed
    Ws_ptr,           # bf16,  (N,)
    Bias_ptr,         # bf16,  (N,) or unused
    Out_ptr,          # bf16,  (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be a multiple of 4
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 4
    K_PACKED = K // 4

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Sub-bit shifts within each packed byte: bit slot j -> shift 2*j.
    # offs_k_chunk[k] = (k // 4) * 4 + (k % 4) maps the BLOCK_K dim to
    # ``(packed_byte_idx, sub_bit_idx)`` pairs.  We construct the unpack
    # via 3D broadcast and reshape, which compiles to plain shift+mask.
    offs_kp = tl.arange(0, BLOCK_K_PACKED)            # (BLOCK_K_PACKED,)
    sub = tl.arange(0, 4).to(tl.uint8)                # (4,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_off in range(0, K_PACKED, BLOCK_K_PACKED):
        k_packed_idx = k_off + offs_kp                # (BLOCK_K_PACKED,)
        k_p_mask = k_packed_idx < K_PACKED            # (BLOCK_K_PACKED,)

        # Load packed weights: (BLOCK_N, BLOCK_K_PACKED) uint8
        wp_ptrs = Wp_ptr + offs_n[:, None] * stride_wn + k_packed_idx[None, :] * stride_wk
        wp = tl.load(
            wp_ptrs,
            mask=n_mask[:, None] & k_p_mask[None, :],
            other=0,
        )                                             # uint8 (BLOCK_N, BLOCK_K_PACKED)

        # Unpack each byte to four int8 ternary values: shape (BLOCK_N, BLOCK_K_PACKED, 4)
        # bit slot j -> ((byte >> 2j) & 3) - 1
        unpacked = ((wp[:, :, None] >> (2 * sub)[None, None, :]) & 3).to(tl.int8) - 1
        # Reshape (BLOCK_N, BLOCK_K_PACKED, 4) -> (BLOCK_N, BLOCK_K) in C order so
        # the rightmost (sub-bit) varies fastest, matching K layout.
        w = tl.reshape(unpacked, (BLOCK_N, BLOCK_K))   # int8 (BLOCK_N, BLOCK_K)

        # Load activations: (BLOCK_M, BLOCK_K) int8.  K indices for this
        # chunk are k_off*4 + [0..BLOCK_K).
        k_idx = k_off * 4 + tl.arange(0, BLOCK_K)      # (BLOCK_K,)
        k_mask = k_idx < K
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0,
        )                                              # int8 (BLOCK_M, BLOCK_K)

        # int8 dot product -> int32 accumulator
        # tl.dot expects (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N).
        acc += tl.dot(a, tl.trans(w), out_dtype=tl.int32)

    # Per-row activation scale (BLOCK_M,) and per-row weight scale (BLOCK_N,).
    #
    # The microsoft/bitnet-b1.58-2B-4T checkpoint is consumed by HF's
    # ``transformers.integrations.bitnet.AutoBitLinear`` (NOT the older
    # ``BitLinear`` in the same file - check the model with
    # ``isinstance(model.model.layers[0].self_attn.q_proj, AutoBitLinear)``,
    # confirmed via ``quantization_config.linear_class == 'autobitlinear'``).
    #
    # AutoBitLinear's forward is::
    #
    #     y = F.linear(fake_quant(x), ternary_weight)
    #     y = y * weight_scale
    #
    # where ``fake_quant(x) = round(x * s).clamp(-128, 127) / s`` recovers the
    # original bf16 magnitude.  Equivalently, with explicit int8 quantization::
    #
    #     y = (x_int @ ternary_weight.T) / act_scale * weight_scale
    #
    # i.e. *multiply* by ``weight_scale`` and *divide* by ``act_scale``.  This
    # is the SOTA convention also used by ``vllm_repo/BitNet``'s CUDA kernel
    # (``bitnet_kernels.cu``: ``red_buf0[0] / s[0] * ws[ws_idx]``).
    #
    # Note: the older ``BitLinear`` class in transformers does ``y / (s * ws)``
    # which would be correct if ``weight_scale`` were the *inverse* of the
    # SOTA weight_scale.  That is *not* the case for this checkpoint - the
    # AutoBitLinear path is what was used in training.
    a_scale = tl.load(A_scale_ptr + offs_m, mask=m_mask, other=1.0).to(tl.float32)
    ws = tl.load(Ws_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    # Match Microsoft BitNet's CUDA kernel order exactly:
    # ``(float)acc / (float)act_scale * (float)weight_scale``.
    # Reordering this as ``acc * weight_scale / act_scale`` changes fp32
    # rounding enough to flip close greedy logits on BitNet.
    out_f = (acc.to(tl.float32) / a_scale[:, None]) * ws[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        out_f = out_f + bias[None, :]

    out = out_f.to(tl.bfloat16)
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Per-token int8 activation quantization (matches SOTA ``quant_input``):
#
#     s = 127 / max|x|  per row
#     q = round(x * s).clamp(-128, 127)   in int8
#
# ``s`` is returned in bf16 so the GEMM kernel can fold it into the output
# scaling step.  Implemented as a single Triton kernel so we get one CUDA
# launch with one HBM read + one HBM write (no fp32 promotion copy, no
# intermediate ``absmax``/``scale`` tensors -- the eager-PyTorch version
# is ~5x slower because it dispatches 4-5 separate kernels).
# ---------------------------------------------------------------------------

@triton.jit
def _activation_quant_int8_kernel(
    X_ptr,            # bf16/fp16/fp32 (M, K)
    Q_ptr,            # int8           (M, K)
    S_ptr,            # bf16           (M,)
    M, K,
    stride_xm, stride_xk,
    stride_qm, stride_qk,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    # One program per row; we sweep ``BLOCK_K`` lanes per iteration.
    # ``BLOCK_K`` is set at launch to the next pow2 >= K so the loop
    # has exactly one iteration for the typical hidden sizes.
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K

    x_ptrs = X_ptr + pid * stride_xm + offs_k * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Per-row absmax in fp32, clamped (matches HF ActQuant).
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-5)
    scale = 127.0 / absmax

    q = tl.extra.cuda.libdevice.rint(x * scale)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)

    q_ptrs = Q_ptr + pid * stride_qm + offs_k * stride_qk
    tl.store(q_ptrs, q, mask=mask)

    tl.store(S_ptr + pid, scale.to(tl.bfloat16))


_bn_lib_q = torch.library.Library("kb_nano_bitnet_q", "DEF")
_bn_lib_q.define(
    "act_quant_int8(Tensor x, Tensor! q_out, Tensor! s_out) -> ()"
)


def _act_quant_impl(x: torch.Tensor, q_out: torch.Tensor,
                    s_out: torch.Tensor) -> None:
    M, K = x.shape
    BLOCK_K = triton.next_power_of_2(K)
    _activation_quant_int8_kernel[(M,)](
        x, q_out, s_out, M, K,
        x.stride(0), x.stride(1),
        q_out.stride(0), q_out.stride(1),
        BLOCK_K=BLOCK_K, num_warps=8 if BLOCK_K >= 4096 else 4,
    )


_bn_lib_q.impl("act_quant_int8", _act_quant_impl, "CUDA")


@torch.library.impl(_bn_lib_q, "act_quant_int8", "Meta")
def _act_quant_meta(x, q_out, s_out):
    pass


def _activation_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 activation quantization (single Triton launch).

    Mirrors HF's ``transformers.integrations.bitnet.ActQuant.forward``:
    promote to fp32, take per-row absmax (clamped at 1e-5), compute
    ``s = 127 / absmax``, quantize via ``round(x * s).clamp(-128, 127)``.

    Returns ``(q_int8, scale_bf16)`` where ``q_int8`` has the same shape
    as ``x`` and ``scale_bf16`` has shape ``(M,)``.
    """
    M, K = x.shape
    q_out = torch.empty_like(x, dtype=torch.int8)
    s_out = torch.empty(M, dtype=torch.bfloat16, device=x.device)
    torch.ops.kb_nano_bitnet_q.act_quant_int8(x, q_out, s_out)
    return q_out, s_out


# ---------------------------------------------------------------------------
# Optional Microsoft ladder decode kernel (M == 1).
# ---------------------------------------------------------------------------

_OFFICIAL_LIB = None
_OFFICIAL_LIB_PATH = None
_OFFICIAL_LIB_FAILED = False


def _get_official_bitnet_lib():
    global _OFFICIAL_LIB, _OFFICIAL_LIB_PATH, _OFFICIAL_LIB_FAILED
    path = os.environ.get("KB_BITNET_KERNEL_LIB")
    if not path:
        return None
    if _OFFICIAL_LIB is not None and _OFFICIAL_LIB_PATH == path:
        return _OFFICIAL_LIB
    if _OFFICIAL_LIB_FAILED:
        return None
    if not os.path.isfile(path):
        _OFFICIAL_LIB_FAILED = True
        return None
    try:
        lib = ctypes.CDLL(path)
        fn = lib.bitlinear_int8xint2
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        fn.restype = None
    except Exception:
        _OFFICIAL_LIB_FAILED = True
        return None
    _OFFICIAL_LIB = lib
    _OFFICIAL_LIB_PATH = path
    return lib


def bitnet_official_kernel_available() -> bool:
    return _get_official_bitnet_lib() is not None


def _official_shape_supported(m: int, n: int, k: int) -> bool:
    return m == 1 and (n, k) in {
        (3840, 2560),
        (2560, 2560),
        (13824, 2560),
        (2560, 6912),
        (4800, 3200),
        (3200, 3200),
        (20480, 3200),
        (3200, 10240),
        (5120, 27648),
        (55296, 5120),
    }


_bn_official_lib = torch.library.Library("kb_nano_bitnet_official", "DEF")
_bn_official_lib.define(
    "int8xint2_gemm(Tensor input_int8, Tensor act_scale, Tensor weight, "
    "Tensor weight_scale, Tensor! output) -> ()"
)


def _official_gemm_impl(
    input_int8: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    m, k = input_int8.shape
    n, k_packed = weight.shape
    assert k == k_packed * 4
    if not _official_shape_supported(m, n, k):
        raise RuntimeError(
            f"Microsoft BitNet ladder kernel does not support "
            f"M={m}, N={n}, K={k}"
        )
    lib = _get_official_bitnet_lib()
    if lib is None:
        raise RuntimeError(
            "KB_BITNET_KERNEL_LIB must point to Microsoft BitNet "
            "gpu/bitnet_kernels/libbitnet.so"
        )
    stream = torch.cuda.current_stream()
    lib.bitlinear_int8xint2(
        ctypes.c_void_p(input_int8.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(act_scale.data_ptr()),
        ctypes.c_void_p(weight_scale.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_void_p(stream.cuda_stream),
    )


_bn_official_lib.impl("int8xint2_gemm", _official_gemm_impl, "CUDA")


@torch.library.impl(_bn_official_lib, "int8xint2_gemm", "Meta")
def _official_gemm_meta(input_int8, act_scale, weight, weight_scale, output):
    pass


def bitnet_int8xint2_linear_official(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Microsoft ladder W1.58A8 decode path for M == 1."""
    assert weight.dtype == torch.int8
    n, k_packed = weight.shape
    k = k_packed * 4
    in_shape = x.shape
    assert in_shape[-1] == k
    x_2d = x.reshape(-1, k).contiguous()
    if not _official_shape_supported(x_2d.shape[0], n, k):
        raise RuntimeError(
            f"Unsupported Microsoft BitNet kernel shape: "
            f"M={x_2d.shape[0]}, N={n}, K={k}"
        )
    q_int8, act_scale = _activation_quant_int8(x_2d)
    out = torch.empty(x_2d.shape[0], n, dtype=torch.bfloat16, device=x.device)
    torch.ops.kb_nano_bitnet_official.int8xint2_gemm(
        q_int8, act_scale, weight, weight_scale, out,
    )
    return out.reshape(*in_shape[:-1], n)


# ---------------------------------------------------------------------------
# torch.library custom op — opaque to torch.compile (mirrors fp8_linear's
# ``kb_nano_fp8::fp8_gemm_nt`` pattern).
# ---------------------------------------------------------------------------

_bn_lib = torch.library.Library("kb_nano_bitnet", "DEF")

_bn_lib.define(
    "int8xint2_gemm(Tensor input_int8, Tensor act_scale, "
    "Tensor weight, Tensor weight_scale, Tensor? bias, Tensor! output) -> ()"
)


def _bitnet_gemm_impl(
    input_int8: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    M, K = input_int8.shape
    N, K_packed = weight.shape
    assert K == K_packed * 4, f"K mismatch: input K={K} vs weight K/4={K_packed}"
    assert weight_scale.numel() == N
    assert act_scale.numel() == M

    # Tile selection (hand-picked from H200 microbench):
    #
    # * Decode (M <= 64): tall narrow tiles. ``BLOCK_K=256`` was the
    #   single biggest win in microbench (-7% vs 64) because it cuts the
    #   K-loop trip count from K/64 -> K/256 and amortizes int8-dot
    #   overhead.  Wider BLOCK_N (128/256) hurts at this M because we
    #   spawn fewer grid blocks than SMs.
    # * Small (64 < M <= 256): same shape, slightly wider warps.
    # * Prefill (M > 256): wide tiles to amortize launch + maximize
    #   tensor-core occupancy.  ``BLOCK_M=64, BLOCK_N=256, BLOCK_K=64,
    #   warps=8`` matches H100/H200 m16n8k32 WMMA mode best.
    if M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 256
        num_warps, num_stages = 4, 3
    elif M <= 256:
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 256
        num_warps, num_stages = 8, 4
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 256, 64
        num_warps, num_stages = 8, 3
    if BLOCK_K > K:
        BLOCK_K = max(4, triton.next_power_of_2(K))
        BLOCK_K = ((BLOCK_K + 3) // 4) * 4

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    bias_ptr = bias if bias is not None else input_int8  # dummy ptr; HAS_BIAS gates loads
    _bitnet_int8xint2_gemm_kernel[grid](
        input_int8, act_scale, weight, weight_scale, bias_ptr, output,
        M, N, K,
        input_int8.stride(0), input_int8.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )


_bn_lib.impl("int8xint2_gemm", _bitnet_gemm_impl, "CUDA")


@torch.library.impl(_bn_lib, "int8xint2_gemm", "Meta")
def _bitnet_gemm_meta(input_int8, act_scale, weight, weight_scale, bias, output):
    pass


# ---------------------------------------------------------------------------
# Public entry point used by ``BitLinear`` / ``BitLinearMerged``.
# ---------------------------------------------------------------------------

def bitnet_int8xint2_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W1.58A8 linear: bf16 activation -> int8 quant -> int8xint2 GEMM -> bf16.

    Args:
        x: ``(..., K)`` bf16/fp16 activation.
        weight: ``(N, K/4)`` uint8 packed ternary weight (KN-packed layout).
        weight_scale: ``(N,)`` bf16 per-output-row dequant scale.
        bias: optional ``(N,)`` bf16 bias added in fp32 inside the kernel.

    Returns:
        ``(..., N)`` bf16 output.
    """
    assert weight.dtype == torch.uint8
    assert weight.ndim == 2
    N, K_packed = weight.shape
    K = K_packed * 4
    in_shape = x.shape
    assert in_shape[-1] == K, f"x last dim {in_shape[-1]} != K {K}"

    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.shape[0]

    q_int8, act_scale = _activation_quant_int8(x_2d)

    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    torch.ops.kb_nano_bitnet.int8xint2_gemm(
        q_int8, act_scale, weight, weight_scale, bias, out,
    )

    return out.reshape(*in_shape[:-1], N)


# ---------------------------------------------------------------------------
# Weight repacking: HF (out/4, in) uint8 -> KN-packed (out, in/4) uint8
# ---------------------------------------------------------------------------

VALUES_PER_BYTE = 4


def unpack_hf_ternary(packed_hf: torch.Tensor) -> torch.Tensor:
    """Unpack HF's (out/4, in) uint8 into int8 (out, in) ternary in {-1,0,+1}.

    Mirrors ``transformers.integrations.bitnet.unpack_weights``: byte at
    (r, c) holds 4 ternary values for output rows ``r + i*(out/4)`` for
    ``i in [0, 1, 2, 3]`` in bits ``2i .. 2i+1``, biased by +1.
    """
    if packed_hf.dtype != torch.uint8:
        packed_hf = packed_hf.to(torch.uint8)
    packed_rows, in_features = packed_hf.shape
    out_rows = packed_rows * VALUES_PER_BYTE
    unpacked = torch.empty((out_rows, in_features), dtype=torch.uint8,
                           device=packed_hf.device)
    for i in range(VALUES_PER_BYTE):
        mask = 3 << (2 * i)
        unpacked[i * packed_rows:(i + 1) * packed_rows] = (packed_hf & mask) >> (2 * i)
    return unpacked.to(torch.int8) - 1


def repack_ternary_kn(unpacked: torch.Tensor) -> torch.Tensor:
    """Repack int8 ternary ``(out, in)`` into uint8 ``(out, in/4)`` (KN layout).

    Each output byte stores 4 consecutive ternary K values for one output
    row, biased by +1.  Bit ``2j..2j+1`` of byte ``(n, c)`` holds the value
    at ``(n, 4c + j)``.
    """
    assert unpacked.dim() == 2
    out_features, in_features = unpacked.shape
    assert in_features % VALUES_PER_BYTE == 0, \
        f"in_features={in_features} not divisible by {VALUES_PER_BYTE}"
    biased = (unpacked + 1).to(torch.int32)            # (out, in), {0,1,2}
    biased = biased.reshape(out_features, in_features // VALUES_PER_BYTE,
                            VALUES_PER_BYTE)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32,
                          device=unpacked.device)
    packed = (biased << shifts).sum(dim=-1).to(torch.uint8)
    return packed.contiguous()


def hf_packed_to_kn_packed(packed_hf: torch.Tensor) -> torch.Tensor:
    """Convenience: HF ``(out/4, in)`` -> KN-packed ``(out, in/4)`` uint8."""
    return repack_ternary_kn(unpack_hf_ternary(packed_hf))


_SOTA_PACK_MAPPING = None


def _sota_pack_mapping() -> np.ndarray:
    global _SOTA_PACK_MAPPING
    if _SOTA_PACK_MAPPING is not None:
        return _SOTA_PACK_MAPPING
    mapping = np.zeros((16, 32, 2), dtype=np.int64)
    for i in range(16):
        for j in range(32):
            thread_id = i * 2 + j // 16
            row = (thread_id // 16) * 8 + (thread_id % 8)
            col = (j % 16) + 16 * ((thread_id % 16) // 8)
            mapping[i, j] = (row, col)
    _SOTA_PACK_MAPPING = mapping
    return mapping


def pack_ternary_sota_ladder(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack ternary ``(N, K)`` weights for Microsoft's M=1 ladder kernel."""
    assert unpacked.dim() == 2
    n, k = unpacked.shape
    assert n % 16 == 0 and k % 32 == 0
    device = unpacked.device
    weight = (unpacked.detach().cpu().numpy().astype(np.int8) + 2).astype(np.int8)
    mapping = _sota_pack_mapping()

    n_blocks = np.arange(n // 16)[:, None, None, None]
    k_blocks = np.arange(k // 32)[None, :, None, None]
    src_n = n_blocks * 16 + mapping[None, None, :, :, 0]
    src_k = k_blocks * 32 + mapping[None, None, :, :, 1]
    permuted = weight[src_n, src_k]

    compressed = np.zeros((n // 16, k // 32, 16, 8), dtype=np.int8)
    for j in range(8):
        for lane in range(4):
            compressed[..., j] |= (
                permuted[..., j * 4 + lane] << (lane * 2)
            ).astype(np.int8)

    qweight = compressed.view(np.int32)
    interleaved = np.zeros_like(qweight)
    for i in range(4):
        for j in range(4):
            offset = i * 4 + j
            shift = (offset % 4) * 8 + (offset // 4) * 2
            interleaved |= ((qweight >> (2 * offset)) & 3) << shift

    packed = interleaved.view(np.int8).reshape(n, k // 4).copy()
    return torch.from_numpy(packed).to(device=device, non_blocking=True)


def unpack_kn_to_ternary(packed_kn: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`repack_ternary_kn`: KN-packed ``(out, in/4)``
    uint8 -> ``(out, in)`` int8 ternary in ``{-1, 0, +1}``.

    Used by ``BitLinear.process_weights_after_loading`` to materialize
    the bf16 fake-quant weight buffer once at load time.  Pure-tensor
    GPU op (4 bit-shift + mask + concat); no kernel launch overhead
    matters here because it runs once per layer.
    """
    assert packed_kn.dtype == torch.uint8
    assert packed_kn.dim() == 2
    out_features, in_packed = packed_kn.shape
    p = packed_kn.to(torch.int32)
    # Decompose each byte into 4 ternary lanes ``(byte >> 2j) & 3 - 1``
    # so the rightmost lane (j=0) is K-position 4c+0, ..., (j=3) is 4c+3.
    lanes = torch.empty((out_features, in_packed, VALUES_PER_BYTE),
                        dtype=torch.int8, device=packed_kn.device)
    for j in range(VALUES_PER_BYTE):
        lanes[..., j] = ((p >> (2 * j)) & 3).to(torch.int8) - 1
    return lanes.reshape(out_features, in_packed * VALUES_PER_BYTE)


# ---------------------------------------------------------------------------
# Module wrapper for use as an L1 op in benchmarks / kernel-swapper plumbing.
# ---------------------------------------------------------------------------

class BitnetInt8xInt2Gemm(nn.Module):
    """Stateless wrapper around the W1.58A8 GEMM custom op."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                weight_scale: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        return bitnet_int8xint2_linear(x, weight, weight_scale, bias)
