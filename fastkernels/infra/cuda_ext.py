"""Per-op CUDA extension loader for L1 baseline kernels.

Each L1 op that needs CUDA owns a single self-contained ``<op>.cu`` beside its
``<op>.py`` (see ``utils/amalgamate_csrc.py`` for how those files are generated).
The op lazily JIT-compiles *only* its own source via :func:`load_op`, producing an
independent extension module -- so vendored helper code inlined into several ops
never collides (separate translation units, separate ``.so`` files).

The first import of an op JIT-compiles its sources. When a build is pending (cold
cache, or a source edited since the last build) we announce it and pass
``verbose=True`` so ninja's per-file ``[k/n]`` progress streams.

Builds are pinned to the *local* GPU architecture. Set
``FASTKERNELS_CUDA_ARCH_LIST`` to override the target list verbatim, or to the
empty string to leave ``TORCH_CUDA_ARCH_LIST`` untouched.
"""

from __future__ import annotations

import inspect
import os
import site
import subprocess

from torch.utils.cpp_extension import _get_build_directory, load as _load_ext


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
    major = arch.split()[0].split(".")[0]
    if major == "10":
        return ["-DENABLE_NVFP4_SM100=1"]
    if major == "12":
        return ["-DENABLE_NVFP4_SM120=1"]
    return []


def cutlass_include() -> str | None:
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


_pin_build_arch()
_ARCH = os.environ.get("TORCH_CUDA_ARCH_LIST")

# Shared flags for every op (mirror the old monolithic csrc/__init__.py build).
_BASE_CUDA_CFLAGS = [
    "-O3",
    "-DENABLE_FP8",
    "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    *_nvfp4_flags(_ARCH),
]
_BASE_CFLAGS = ["-O3", "-DENABLE_FP8"]
_BASE_LDFLAGS = ["-lcublas", "-lcuda"]


def _build_pending(name: str, src_paths: list[str]) -> bool:
    try:
        so_path = os.path.join(
            _get_build_directory(name, verbose=False), f"{name}.so")
    except Exception:
        return True
    if not os.path.exists(so_path):
        return True
    so_mtime = os.path.getmtime(so_path)
    return any(os.path.getmtime(s) > so_mtime for s in src_paths)


def _load_op_at(
    base: str,
    name: str,
    sources: list[str],
    *,
    need_cutlass: bool = False,
    extra_cuda_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
):
    src_paths = [os.path.join(base, s) for s in sources]

    include_paths = [base]
    if need_cutlass:
        c = cutlass_include()
        if c:
            include_paths.append(c)

    cuda_cflags = list(_BASE_CUDA_CFLAGS)
    if extra_cuda_cflags:
        cuda_cflags += extra_cuda_cflags

    verbose = _build_pending(name, src_paths)
    if verbose:
        print(f"[fastkernels] Building CUDA op {name!r} ({len(src_paths)} "
              f"source(s)) for arch "
              f"{os.environ.get('TORCH_CUDA_ARCH_LIST', 'auto')!r}"
              f" -- one-time JIT compile; streaming ninja progress ...",
              flush=True)

    return _load_ext(
        name=name,
        sources=src_paths,
        extra_include_paths=include_paths,
        extra_cuda_cflags=cuda_cflags,
        extra_cflags=list(_BASE_CFLAGS),
        extra_ldflags=list(extra_ldflags or _BASE_LDFLAGS),
        verbose=verbose,
    )


def load_op(name: str, sources: str | list[str], **kwargs):
    """JIT-load and return a single op's CUDA extension immediately.

    ``sources`` are resolved relative to the calling module's directory, so an op
    just names its sidecar file(s): ``load_op("moe_sum", "moe_sum.cu")``.
    """
    if isinstance(sources, str):
        sources = [sources]
    base = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    return _load_op_at(base, name, sources, **kwargs)


class _LazyExt:
    """Lazy handle to a per-op extension.

    Behaves like the compiled module (``handle.some_symbol(...)``) but defers the
    JIT build until the first attribute access, and caches it thereafter. Lets op
    modules keep call sites like ``_C.moe_sum(...)`` and ``hasattr(_C, ...)``.
    """

    def __init__(self, base: str, name: str, sources: list[str], **kwargs):
        self.__dict__["_spec"] = (base, name, sources, kwargs)
        self.__dict__["_mod"] = None

    def _load(self):
        if self.__dict__["_mod"] is None:
            base, name, sources, kwargs = self.__dict__["_spec"]
            self.__dict__["_mod"] = _load_op_at(base, name, sources, **kwargs)
        return self.__dict__["_mod"]

    def __getattr__(self, item):
        return getattr(self._load(), item)


def lazy_op(name: str, sources: str | list[str], **kwargs) -> _LazyExt:
    """Return a lazy handle to an op extension, resolved relative to the caller.

    Usage in an L1 op module::

        from ....infra.cuda_ext import lazy_op
        _C = lazy_op("moe_sum", "moe_sum.cu")
        ...
        _C.moe_sum(x, out)   # builds on first use
    """
    if isinstance(sources, str):
        sources = [sources]
    base = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    return _LazyExt(base, name, sources, **kwargs)
