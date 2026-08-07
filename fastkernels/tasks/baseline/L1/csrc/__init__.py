"""Shared CUDA extension loader for L1 baseline kernels.

Compiles and caches the extension once; all task modules import _C from here.

The first import JIT-compiles the CUDA sources. When a build is pending (cold
cache, or a source edited since the last build) we announce it and pass
``verbose=True`` so ninja's per-file ``[k/n]`` progress streams.

We also pin that build to the *local* GPU architecture. Set
``FASTKERNELS_CUDA_ARCH_LIST`` to override the target list verbatim, or to the
empty string to leave ``TORCH_CUDA_ARCH_LIST`` untouched.
"""

from __future__ import annotations

import os
import site
import subprocess

from torch.utils.cpp_extension import _get_build_directory, load as _load_ext

_DIR = os.path.dirname(os.path.abspath(__file__))
_NAME = "fastkernels_L1_ops"

# Vendored vLLM 0.26 CUDA kernels (verbatim device code; host wrappers adapted
# from torch::stable::Tensor -> torch::Tensor). See vllm_port/.
_VLLM_PORT = [
    "vllm_port/libtorch_stable/layernorm_kernels.cu",
    "vllm_port/libtorch_stable/activation_kernels.cu",
    "vllm_port/libtorch_stable/pos_encoding_kernels.cu",
    "vllm_port/libtorch_stable/cache_kernels.cu",
    "vllm_port/libtorch_stable/nvfp4_kv_cache_kernels.cu",
    "vllm_port/libtorch_stable/cuda_utils_kernels.cu",
    "vllm_port/libtorch_stable/quantization/w8a8/fp8/common.cu",
    "vllm_port/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu",
    "vllm_port/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu",
    "vllm_port/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu",
    "vllm_port/libtorch_stable/cutlass_extensions/common.cpp",
    "vllm_port/libtorch_stable/sampler.cu",
    "vllm_port/libtorch_stable/topk.cu",
    "vllm_port/libtorch_stable/cooperative_topk.cu",
    "vllm_port/libtorch_stable/attention/merge_attn_states.cu",
    "vllm_port/libtorch_stable/mamba/selective_scan_fwd.cu",
]

_SOURCES = [
    "binding.cpp", "rmsnorm.cu", "rmsnorm_quant.cu",
    # RoPE: vLLM's pos_encoding_kernels.cu (below in _VLLM_PORT).
    "moe_sum.cu", "moe_align.cu", "moe_topk_softmax.cu",
    "eagle_utils.cu",
    "dsv3_router_gemm_entry.cu",
    "dsv3_router_gemm_float_out.cu",
    "dsv3_router_gemm_bf16_out.cu",
    "router_gemm_bf16_fp32.cu",
    "grouped_topk_kernels.cu",
] + _VLLM_PORT
_SOURCE_PATHS = [os.path.join(_DIR, f) for f in _SOURCES]


def _cutlass_include() -> str | None:
    try:
        import flashinfer.data  # type: ignore
        base = os.path.join(
            os.path.dirname(flashinfer.data.__file__), "cutlass", "include")
        if os.path.isdir(base):
            return base
    except Exception:
        pass
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for rel in (
            "flashinfer/data/cutlass/include",
            "cutlass_library/source/include",
            "tilelang/3rdparty/cutlass/include",
            "deep_gemm/include",
        ):
            p = os.path.join(sp, rel)
            if os.path.isdir(p):
                return p
    return None


def _build_pending() -> bool:
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
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    # NVFP4 / Blackwell family features need the 'a' (architecture-specific)
    # variant; plain 10.0 rejects e2m1x2 in ptxas.
    mapped = []
    for c in caps:
        major = c.split(".")[0]
        if major in ("9", "10", "12"):
            mapped.append(f"{c}a" if not c.endswith("a") else c)
        else:
            mapped.append(c)
    return " ".join(mapped) or None


def _pin_build_arch() -> None:
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    arch = _local_cuda_arch()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch


def _nvfp4_flags(arch: str | None) -> list[str]:
    """Enable the matching NVFP4 quant kernel for the local arch."""
    if not arch:
        return []
    # Take the first (primary) arch token, e.g. "10.0" from "10.0" / "10.0 9.0".
    major = arch.split()[0].split(".")[0]
    if major == "10":
        return ["-DENABLE_NVFP4_SM100=1"]
    if major == "12":
        return ["-DENABLE_NVFP4_SM120=1"]
    return []


_pin_build_arch()
_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
_EXTRA_INCLUDE = [os.path.join(_DIR, "vllm_port")]
_cutlass = _cutlass_include()
if _cutlass:
    _EXTRA_INCLUDE.append(_cutlass)

_verbose = _build_pending()
if _verbose:
    print(f"[fastkernels] Building CUDA extension {_NAME!r} ({len(_SOURCES)} "
          f"sources) for arch {os.environ.get('TORCH_CUDA_ARCH_LIST', 'auto')!r}"
          f" -- one-time JIT compile; streaming ninja progress ...", flush=True)

_C = _load_ext(
    name=_NAME,
    sources=_SOURCE_PATHS,
    extra_include_paths=_EXTRA_INCLUDE,
    extra_cuda_cflags=[
        "-O3",
        "-DENABLE_FP8",
        "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--expt-extended-lambda",
        *_nvfp4_flags(_arch),
    ],
    extra_cflags=["-O3", "-DENABLE_FP8"],
    extra_ldflags=["-lcublas", "-lcuda"],
    verbose=_verbose,
)
