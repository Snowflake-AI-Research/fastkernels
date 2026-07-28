"""Provision reference libraries and heavy task kernels for ``validate``.

The mandatory ``dependencies`` in ``pyproject.toml`` are the *runtime* set:
everything the fastkernels engine, candidate kernels, and the e2e eval harness
import to run. This module installs the *reference-only* extras — and the two
things pip cannot express, git checkouts and source builds — that the
``validate`` phase needs to instantiate the reference implementations it
compares against.

Everything installed here is idempotent and lands in the active Python
environment (``sys.executable``) or under ``THIRD_PARTY_DIR``
(``~/.fastkernels/third_party`` by default, override with
``FASTKERNELS_THIRD_PARTY_DIR``).

Usage::

    python -m fastkernels.validate.provision --list
    python -m fastkernels.validate.provision --all
    python -m fastkernels.validate.provision ttt dp3 3dgs

or, driven by the validate dispatcher, ``fastkernels validate <table> --provision``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastkernels import THIRD_PARTY_DIR

# --- Checkout / build locations -------------------------------------------
TTT_DIR = THIRD_PARTY_DIR / "ttt_e2e"
DP3_DIR = THIRD_PARTY_DIR / "3D-Diffusion-Policy"
# bench_dp3 imports from the inner package directory of the nested repo.
DP3_INNER = DP3_DIR / "3D-Diffusion-Policy"
NGP_DIR = THIRD_PARTY_DIR / "instant-ngp"
FBGEMM_DIR = THIRD_PARTY_DIR / "FBGEMM"
PTV3_DIR = THIRD_PARTY_DIR / "PointTransformerV3"
FASTDLLM_DIR = THIRD_PARTY_DIR / "Fast-dLLM"


# --- Shell helpers ---------------------------------------------------------
def _run(cmd, *, cwd: Path | None = None, env: dict | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    where = f"  (cwd={cwd})" if cwd else ""
    print(f"    $ {printable}{where}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, check=True)


def _pip(*reqs: str) -> None:
    _run([sys.executable, "-m", "pip", "install", *reqs])


def _importable(module: str, *, extra_path: str | None = None) -> bool:
    """True if ``module`` imports in a *fresh* interpreter (the target env).

    ``extra_path`` is prepended to ``sys.path`` first, so a reference repo that
    is not pip-installed (e.g. a git checkout) can be import-checked.
    """
    prelude = f"import sys; sys.path.insert(0, {extra_path!r}); " if extra_path else ""
    return (
        subprocess.run(
            [sys.executable, "-c", f"{prelude}import {module}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _git_clone(url: str, dest: Path, *, recursive: bool = False, depth: int = 1) -> None:
    if dest.exists():
        print(f"    [skip] already present: {dest}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth)]
    if recursive:
        cmd += ["--recurse-submodules", "--shallow-submodules"]
    cmd += [url, str(dest)]
    _run(cmd)


def _torch_cc() -> tuple[int, int]:
    """(major, minor) compute capability of GPU 0, e.g. (10, 0) for B200."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch;cc=torch.cuda.get_device_capability();print(cc[0],cc[1])",
        ],
        capture_output=True,
        text=True,
    )
    try:
        major, minor = out.stdout.split()
        return int(major), int(minor)
    except ValueError:
        return (10, 0)


# --- Per-component provisioning steps --------------------------------------
def _prov_ttt() -> None:
    _pip("jax[cuda12]", "equinox>=0.11.12", "optax>=0.2.4", "jaxtyping>=0.2.36", "hydra-core>=1.3")
    _git_clone("https://github.com/test-time-training/e2e", TTT_DIR)


def _check_ttt() -> bool:
    return (TTT_DIR / "ttt" / "model" / "transformer.py").is_file() and _importable("jax")


def _prov_dp3() -> None:
    # Pure-Python deps of the reference diffusion_policy_3d import chain
    # (omegaconf drives configs, pyarrow/zarr read the dataset + replay buffer,
    # dill loads the reference checkpoints).
    _pip("omegaconf", "pyarrow", "dill", "zarr")
    _git_clone("https://github.com/YanjieZe/3D-Diffusion-Policy", DP3_DIR)
    # diffusion_policy_3d/policy/dp3.py hard-imports pytorch3d.ops at module
    # top level, so the reference cannot be built without it. No wheel matches
    # recent torch/CUDA — build from source against the installed torch.
    _build_pytorch3d()


def _build_pytorch3d() -> None:
    if _importable("pytorch3d"):
        print("    [skip] pytorch3d already importable", flush=True)
        return
    cc = _torch_cc()
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": f"{cc[0]}.{cc[1]}"}
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "git+https://github.com/facebookresearch/pytorch3d.git",
        ],
        env=env,
    )


def _check_dp3() -> bool:
    # The real bar: the reference DP3 policy must import end to end (pulls in
    # zarr + pytorch3d), not merely that the package directory exists.
    return DP3_INNER.is_dir() and _importable(
        "diffusion_policy_3d.policy.dp3", extra_path=str(DP3_INNER)
    )


def _prov_3dgs() -> None:
    # gsplat JIT-compiles its CUDA extension on first import; install is quick.
    _pip("gsplat")


def _check_3dgs() -> bool:
    return _importable("gsplat")


def _prov_pointcloud() -> None:
    # Candidate path: spconv sparse-conv kernels. No cu13x wheel exists; the
    # cu126 wheel executes on newer CUDA / Blackwell. ``addict`` is imported by
    # the reference model.py (torch_scatter is shimmed by the loader; timm /
    # flash-attn are already runtime deps).
    _pip("spconv-cu126", "addict")
    # Reference path: the official PointTransformerV3 model class is loaded from
    # this checkout (bench_pointcloud runs with reference comparison by default).
    _git_clone("https://github.com/Pointcept/PointTransformerV3", PTV3_DIR)


def _check_pointcloud() -> bool:
    return _importable("spconv.pytorch") and (PTV3_DIR / "model.py").is_file()


def _prov_recsys() -> None:
    # LightGCN reference — pure-Python, reliable.
    _pip("torch-geometric")
    # DLRMv2 reference (torchrec) needs fbgemm_gpu matched to the installed
    # torch. No prebuilt wheel exists for recent torch, so build from source.
    # Best-effort: LightGCN already works if this leg fails.
    try:
        _build_fbgemm()
        # Install torchrec without deps so it cannot pull a mismatched fbgemm
        # wheel over the source build; add its remaining pure deps explicitly.
        _pip("tensordict", "torchmetrics", "pyre-extensions")
        _pip("--no-deps", "torchrec")
    except subprocess.CalledProcessError as exc:
        print(
            f"    WARNING: torchrec/fbgemm_gpu source build failed ({exc}); "
            "DLRMv2 reference unavailable, LightGCN reference still works.",
            flush=True,
        )


def _check_recsys() -> bool:
    # Minimum bar is LightGCN; DLRMv2 (torchrec) is best-effort on top.
    return _importable("torch_geometric")


def _build_fbgemm() -> None:
    if _importable("fbgemm_gpu"):
        print("    [skip] fbgemm_gpu already importable", flush=True)
        return
    _git_clone("https://github.com/pytorch/FBGEMM", FBGEMM_DIR, recursive=True)
    fbdir = FBGEMM_DIR / "fbgemm_gpu"
    cc = _torch_cc()
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": f"{cc[0]}.{cc[1]}"}
    reqs = fbdir / "requirements.txt"
    if reqs.is_file():
        _run([sys.executable, "-m", "pip", "install", "-r", str(reqs)])
    # FBGEMM's setup.py takes --build-target (default = the fbgemm_gpu package
    # torchrec's DLRM needs) and --build-variant (cuda), consuming them before
    # delegating the rest to setuptools' install command.
    _run(
        [
            sys.executable,
            "setup.py",
            "install",
            "--build-target",
            "default",
            "--build-variant",
            "cuda",
        ],
        cwd=fbdir,
        env=env,
    )


def _prov_dllm() -> None:
    # lm_eval supplies the official task prompts (HumanEval, etc.). 0.4.12 tracks
    # transformers 5.x (older builds reference the removed AutoModelForVision2Seq
    # at import time). Fast-dLLM is the official reference for the fastdllm-*
    # baselines; bench_dllm imports its LLaDA code from the v1/ subtree.
    _pip("lm_eval>=0.4.12")
    _git_clone("https://github.com/NVlabs/Fast-dLLM", FASTDLLM_DIR)


def _check_dllm() -> bool:
    return _importable("lm_eval.models.hf_vlms") and (
        FASTDLLM_DIR / "v1" / "llada" / "model" / "configuration_llada.py"
    ).is_file()


def _prov_omni() -> None:
    # vllm-omni itself is a runtime dependency (installed with the package). The
    # only extra the omni references need is s3tokenizer, which backs CosyVoice3
    # speech-token extraction (upstream moved this out of cosyvoice3.utils into an
    # on-GPU S3Tokenizer) and is not declared by vllm-omni.
    _pip("s3tokenizer")


def _check_omni() -> bool:
    return _importable("s3tokenizer") and _importable("vllm_omni")


def _prov_instant_ngp() -> None:
    _git_clone("https://github.com/NVlabs/instant-ngp", NGP_DIR, recursive=True)
    build = NGP_DIR / "build"
    cc = _torch_cc()
    tcnn_arch = f"{cc[0]}{cc[1]}"  # (10, 0) -> "100"
    _run(
        [
            "cmake",
            str(NGP_DIR),
            "-B",
            str(build),
            "-DNGP_BUILD_WITH_GUI=OFF",
            "-DNGP_BUILD_WITH_OPTIX=OFF",
            "-DNGP_BUILD_WITH_VULKAN=OFF",
            f"-DTCNN_CUDA_ARCHITECTURES={tcnn_arch}",
        ]
    )
    _run(["cmake", "--build", str(build), "--config", "RelWithDebInfo", "-j"])


def _check_instant_ngp() -> bool:
    build = NGP_DIR / "build"
    return build.is_dir() and any(build.glob("pyngp*.so"))


# --- Component registry ----------------------------------------------------
@dataclass(frozen=True)
class Component:
    name: str
    summary: str
    harnesses: tuple[str, ...]
    provision: Callable[[], None]
    check: Callable[[], bool]


COMPONENTS: dict[str, Component] = {
    c.name: c
    for c in (
        Component("ttt", "ttt-e2e JAX/Equinox reference + repo", ("bench_ttt_e2e",), _prov_ttt, _check_ttt),
        Component("dp3", "3D-Diffusion-Policy reference + repo", ("bench_dp3",), _prov_dp3, _check_dp3),
        Component("3dgs", "gsplat CUDA kernels (candidate path)", ("bench_3dgs",), _prov_3dgs, _check_3dgs),
        Component("pointcloud", "spconv kernels + PointTransformerV3 reference repo", ("bench_pointcloud",), _prov_pointcloud, _check_pointcloud),
        Component("recsys", "torch-geometric + torchrec references", ("bench_recsys",), _prov_recsys, _check_recsys),
        Component("dllm", "LLaDA lm_eval tasks + Fast-dLLM reference repo", ("bench_dllm",), _prov_dllm, _check_dllm),
        Component("omni", "vllm-omni references (FLUX/HunyuanVideo/CosyVoice3) + s3tokenizer", ("bench_vllm_omni",), _prov_omni, _check_omni),
        Component("instant_ngp", "instant-ngp pyngp source build + fox scene", ("bench_instantngp",), _prov_instant_ngp, _check_instant_ngp),
    )
}

# harness name -> component name
_HARNESS_TO_COMPONENT: dict[str, str] = {
    harness: comp.name for comp in COMPONENTS.values() for harness in comp.harnesses
}


def components_for_harnesses(harnesses) -> list[str]:
    """Component names required by the given validate harnesses (deduped, ordered)."""
    seen: dict[str, None] = {}
    for harness in harnesses:
        name = _HARNESS_TO_COMPONENT.get(harness)
        if name is not None:
            seen.setdefault(name, None)
    return list(seen)


def provision(names, *, force: bool = False) -> int:
    """Provision the named components. Returns the count that failed."""
    failed = 0
    for name in names:
        comp = COMPONENTS.get(name)
        if comp is None:
            print(f"[provision] unknown component {name!r} (known: {', '.join(COMPONENTS)})")
            failed += 1
            continue
        if not force and comp.check():
            print(f"[provision] {name}: already provisioned — skipping")
            continue
        print(f"[provision] {name}: {comp.summary}")
        try:
            comp.provision()
        except Exception as exc:  # noqa: BLE001 - one component must not kill the rest
            print(f"[provision] {name}: FAILED — {exc}")
            failed += 1
            continue
        status = "OK" if comp.check() else "provision ran but check still failing"
        print(f"[provision] {name}: {status}")
        if status != "OK":
            failed += 1
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastkernels.validate.provision",
        description="Install reference libraries / task kernels for the validate phase.",
    )
    parser.add_argument("components", nargs="*", help="Component names to provision (default: all).")
    parser.add_argument("--all", action="store_true", help="Provision every known component.")
    parser.add_argument("--list", action="store_true", help="List components and their status, then exit.")
    parser.add_argument("--force", action="store_true", help="Re-provision even if the check passes.")
    args = parser.parse_args(argv)

    if args.list:
        print(f"third-party dir: {THIRD_PARTY_DIR}")
        for comp in COMPONENTS.values():
            state = "provisioned" if comp.check() else "MISSING"
            print(f"  {comp.name:14s} [{state:11s}] {comp.summary}  (harnesses: {', '.join(comp.harnesses)})")
        return 0

    names = list(COMPONENTS) if (args.all or not args.components) else args.components
    failed = provision(names, force=args.force)
    if failed:
        print(f"\n[provision] {failed} component(s) failed.")
        return 1
    print("\n[provision] all requested components ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
