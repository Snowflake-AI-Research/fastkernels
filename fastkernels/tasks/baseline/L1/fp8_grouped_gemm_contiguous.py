"""DeepGEMM grouped FP8 GEMM for MoE expert execution.

Thin L1 wrapper around ``deep_gemm.m_grouped_fp8_gemm_nt_contiguous``
plus the ``get_mk_alignment_for_contiguous_layout`` helper.  Used by the
DeepSeek MoE FP8 path, which does its own permute/quantize/unpermute via
other L1 ops and calls this module for the two expert GEMMs.

Unlike ``fp8_moe_grouped_gemm.py``, this module does NOT bundle permute
/ gather / scatter logic — it is a straight single-kernel primitive.

Mirrors vLLM's ``disable_ue8m0_cast`` plumbing: that argument is
``not is_deep_gemm_e8m0_used()`` (see
``vllm/utils/deep_gemm.py:206-237``), where E8M0 is enabled iff
DeepGEMM is supported on the current arch (Hopper / Blackwell) AND
``VLLM_USE_DEEP_GEMM_E8M0`` is non-zero (default ``1``).
"""

from __future__ import annotations

import functools
import importlib
import os

import torch
import torch.nn as nn


def _resolve_deep_gemm():
    """Prefer a DeepGEMM build that actually exports kernels.

    Some site-packages installs are a header-only namespace (no ``_C``,
    no ``fp8_gemm_nt``). vLLM vendors a complete copy under
    ``vllm.third_party.deep_gemm``; use that when the top-level package
    is a stub.
    """
    try:
        mod = importlib.import_module("deep_gemm")
        if getattr(mod, "fp8_gemm_nt", None) is not None:
            return mod
    except ImportError:
        mod = None
    try:
        vendored = importlib.import_module("vllm.third_party.deep_gemm")
        if getattr(vendored, "fp8_gemm_nt", None) is not None:
            return vendored
    except Exception:
        pass
    return mod


# DeepGEMM is required for the FP8 path on Hopper+.
deep_gemm = _resolve_deep_gemm()


@functools.cache
def _is_deep_gemm_supported() -> bool:
    """Mirror ``vllm.utils.deep_gemm.is_deep_gemm_supported``.

    Conditions:
    1. ``VLLM_USE_DEEP_GEMM`` env var is non-zero (default 1).
    2. GPU is Hopper (SM90) or Blackwell-class (SM100/SM10x).
    """
    if not bool(int(os.environ.get("VLLM_USE_DEEP_GEMM", "1"))):
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major in (9, 10)


@functools.cache
def _is_deep_gemm_e8m0_used() -> bool:
    """Match vLLM's ``is_deep_gemm_e8m0_used`` exactly
    (``vllm/utils/deep_gemm.py:80-104``).
    """
    if not _is_deep_gemm_supported():
        return False
    if deep_gemm is None or getattr(deep_gemm, "fp8_gemm_nt", None) is None:
        return False
    return bool(int(os.environ.get("VLLM_USE_DEEP_GEMM_E8M0", "1")))


_CACHED_ALIGNMENT: int | None = None


def _resolve_alignment() -> int:
    """Resolve DeepGEMM's M/K alignment for the contiguous layout once.

    The value depends on the hardware (tuned alignment tables) and never
    changes within a process, so caching it avoids calling into the
    untraceable pybind helper on every forward pass (which would otherwise
    graph-break ``torch.compile``).
    """
    global _CACHED_ALIGNMENT
    if _CACHED_ALIGNMENT is None:
        _CACHED_ALIGNMENT = int(
            deep_gemm.get_mk_alignment_for_contiguous_layout()
        )
    return _CACHED_ALIGNMENT


# ---------------------------------------------------------------------------
# Wrap deep_gemm.m_grouped_fp8_gemm_nt_contiguous as an opaque torch.library
# custom op so ``torch.compile`` (Dynamo) does not try to trace into the
# pybind-bound C++ call. This mirrors the pattern used for deep_gemm.fp8_gemm_nt
# in ``fp8_linear.py``.
# ---------------------------------------------------------------------------

_gemm_lib = torch.library.Library("fastkernels_fp8_moe", "DEF")

_gemm_lib.define(
    "m_grouped_gemm_nt_contiguous(Tensor a_fp8, Tensor a_scale, "
    "Tensor b_fp8, Tensor b_scale, Tensor! c_bf16, Tensor expert_ids, "
    "bool disable_ue8m0_cast) -> ()"
)


def _m_grouped_gemm_nt_contiguous_impl(
    a_fp8, a_scale, b_fp8, b_scale, c_bf16, expert_ids, disable_ue8m0_cast,
):
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_fp8, a_scale),
        (b_fp8, b_scale),
        c_bf16,
        expert_ids,
        disable_ue8m0_cast=disable_ue8m0_cast,
    )


_gemm_lib.impl(
    "m_grouped_gemm_nt_contiguous",
    _m_grouped_gemm_nt_contiguous_impl,
    "CUDA",
)


@torch.library.impl(_gemm_lib, "m_grouped_gemm_nt_contiguous", "Meta")
def _m_grouped_gemm_nt_contiguous_meta(
    a_fp8, a_scale, b_fp8, b_scale, c_bf16, expert_ids, disable_ue8m0_cast,
):
    # In-place write to c_bf16; nothing to return.
    pass


class Fp8GroupedGemmContiguous(nn.Module):
    """Grouped FP8 GEMM: ``(A_fp8, a_scale) @ (B_fp8, b_scale) -> C_bf16``.

    The A tensors are in expert-contiguous layout (``[m_sum, K]``), B is
    ``[num_experts, N, K]`` FP8 with DeepGEMM block scales, and
    ``expert_ids`` maps each row in A to an expert index (``-1`` for padding).
    The output ``C_bf16`` is ``[m_sum, N]`` BF16 and is written in-place.
    """

    def __init__(self) -> None:
        super().__init__()
        # Cache the DeepGEMM alignment at construction so ``torch.compile``
        # never sees the pybind-bound lookup during tracing.
        self._alignment: int = _resolve_alignment()
        # ``disable_ue8m0_cast`` is fixed at process start (depends on env +
        # GPU arch) — capture once so the per-call dispatch stays cheap.
        self._disable_ue8m0_cast: bool = not _is_deep_gemm_e8m0_used()

    def alignment(self) -> int:
        """DeepGEMM's required M/K alignment for the contiguous layout."""
        return self._alignment

    def forward(
        self,
        a_fp8: torch.Tensor,
        a_scale: torch.Tensor,
        b_fp8: torch.Tensor,
        b_scale: torch.Tensor,
        c_bf16: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> None:
        torch.ops.fastkernels_fp8_moe.m_grouped_gemm_nt_contiguous(
            a_fp8, a_scale, b_fp8, b_scale, c_bf16, expert_ids,
            self._disable_ue8m0_cast,
        )
