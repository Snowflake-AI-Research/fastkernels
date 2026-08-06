"""Shared CUDA extension loader for L1 baseline kernels.

Compiles and caches the extension once; all task modules import _C from here.

The first import JIT-compiles ~15 CUDA sources (~2 min). That build is otherwise
silent -- capture/bench appear to hang before doing anything -- so when a build
is actually pending (cold cache, or a source edited since the last build) we
announce it and pass ``verbose=True`` so ninja's per-file ``[k/n]`` progress
streams. A warm cache stays completely silent.

We also pin that build to the *local* GPU architecture. An inherited multi-arch
``TORCH_CUDA_ARCH_LIST`` (e.g. ``7.5 8.0 8.6 9.0 10.0 12.0+PTX``) makes nvcc
compile every source once per arch (~6x the work) even though this JIT'd
extension only ever runs on the machine that built it. Set
``FASTKERNELS_CUDA_ARCH_LIST`` to override the target list verbatim (e.g.
``"9.0 10.0"`` to force multi-arch), or to the empty string to leave
``TORCH_CUDA_ARCH_LIST`` untouched.
"""

import os
import subprocess

from torch.utils.cpp_extension import _get_build_directory, load as _load_ext

_DIR = os.path.dirname(os.path.abspath(__file__))
_NAME = "fastkernels_L1_ops"

_SOURCES = [
    "binding.cpp", "rmsnorm.cu", "rmsnorm_quant.cu",
    "pos_enc.cu",
    "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
    "eagle_utils.cu",
    # DeepSeek-V3 router ops (verbatim port of vLLM csrc/moe sources;
    # see binding.cpp for op-level descriptions).
    "dsv3_router_gemm_entry.cu",
    "dsv3_router_gemm_float_out.cu",
    "dsv3_router_gemm_bf16_out.cu",
    "router_gemm_bf16_fp32.cu",
    "grouped_topk_kernels.cu",
]
_SOURCE_PATHS = [os.path.join(_DIR, f) for f in _SOURCES]


def _build_pending() -> bool:
    """Whether importing this module will trigger a (slow) JIT recompile.

    ``True`` when the compiled ``.so`` is missing (cold cache) or any source is
    newer than it (an edited kernel). Best-effort: used only to decide whether
    to show build progress -- ``cpp_extension.load`` still makes the
    authoritative content-hash decision. Errs toward ``True`` (show progress)
    if the build directory can't be resolved.
    """
    try:
        so_path = os.path.join(
            _get_build_directory(_NAME, verbose=False), f"{_NAME}.so")
    except Exception:
        return True
    if not os.path.exists(so_path):
        return True
    so_mtime = os.path.getmtime(so_path)
    return any(os.path.getmtime(s) > so_mtime for s in _SOURCE_PATHS)


def _local_cuda_arch() -> str | None:
    """Compute capability/ies of the visible GPU(s), e.g. ``"10.0"``.

    Read from ``nvidia-smi`` so no CUDA context is created at import time.
    Multiple distinct arches are space-joined (mixed-GPU boxes). ``None`` if it
    can't be determined (no GPU / nvidia-smi unavailable).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    return " ".join(caps) or None


def _pin_build_arch() -> None:
    """Point ``TORCH_CUDA_ARCH_LIST`` at the local GPU arch for the pending build.

    ``FASTKERNELS_CUDA_ARCH_LIST`` takes precedence: a non-empty value is used
    verbatim (force a specific/multi-arch build); an empty value leaves
    ``TORCH_CUDA_ARCH_LIST`` untouched (restore stock torch auto-detection).
    """
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    arch = _local_cuda_arch()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch


# Pin the local GPU arch for the build UNCONDITIONALLY. torch's JIT compile-cache
# key includes TORCH_CUDA_ARCH_LIST, so pinning it only when a build already
# looks pending would flip the key between runs (pinned vs inherited multi-arch)
# and force torch to rebuild every time the key changes. Doing it on every import
# keeps the key stable: build once, then always a cache hit.
_pin_build_arch()
_verbose = _build_pending()
if _verbose:
    print(f"[fastkernels] Building CUDA extension {_NAME!r} ({len(_SOURCES)} "
          f"sources) for arch {os.environ.get('TORCH_CUDA_ARCH_LIST', 'auto')!r}"
          f" -- one-time JIT compile; streaming ninja progress ...", flush=True)

_C = _load_ext(
    name=_NAME,
    sources=_SOURCE_PATHS,
    extra_cuda_cflags=["-O3",
                       "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16",
                       # vLLM's CMake unsets these so its noaux_tc grouped-topk
                       # kernel (ported verbatim into ``grouped_topk_kernels.cu``)
                       # can rely on implicit ``half``/``__nv_bfloat16``<->``float``
                       # constructors.  ``torch.utils.cpp_extension`` defines them
                       # by default; we undefine them here to match vLLM.
                       "-U__CUDA_NO_HALF_OPERATORS__",
                       "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
    verbose=_verbose,
)
