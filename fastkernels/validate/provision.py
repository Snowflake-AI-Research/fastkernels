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
``FASTKERNELS_THIRD_PARTY_DIR``). Two components need their own
interpreters because their torch/CUDA pins conflict with the main env:
``openpi`` (a ``uv`` venv under ``THIRD_PARTY_DIR``) and ``sglang`` (a
conda env named ``sglang-bench``).

Usage::

    python -m fastkernels.validate.provision --list
    python -m fastkernels.validate.provision --all
    python -m fastkernels.validate.provision ttt dp3 3dgs

or automatically by the validate dispatcher: ``fastkernels validate <table>``
provisions the components its scenarios need before dispatching (there is no
``--provision`` flag; only ``--dry-run`` skips it).
"""

from __future__ import annotations

import argparse
import os
import shutil
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
# The Microsoft BitNet GPU reference is a source checkout plus a hand-built CUDA
# kernel and a two-step checkpoint conversion; the harness only ever loads the
# int2/fp16 splits, not the bf16 safetensors they are derived from.
BITNET_DIR = THIRD_PARTY_DIR / "BitNet"
BITNET_GPU_DIR = BITNET_DIR / "gpu"
BITNET_KERNEL_SO = BITNET_GPU_DIR / "bitnet_kernels" / "libbitnet.so"
BITNET_CKPT_DIR = BITNET_GPU_DIR / "checkpoints"
# The OpenPI reference needs its own interpreter: it pins jax[cuda12]==0.5.3 and
# torch==2.7.1, which cannot coexist with the fastkernels env's torch. `uv sync`
# builds that venv inside the checkout; bench_openpi drives it via
# --reference-python. Converted PyTorch checkpoints land next to it.
OPENPI_DIR = THIRD_PARTY_DIR / "openpi"
OPENPI_VENV_PYTHON = OPENPI_DIR / ".venv" / "bin" / "python"
OPENPI_ASSETS_DIR = THIRD_PARTY_DIR / "openpi-assets"
# openpi asset name -> the JAX checkpoint it is downloaded from. Each is
# converted once to `OPENPI_ASSETS_DIR/<name>_pytorch` (model.safetensors +
# config.json), which is the only layout both sides of the benchmark can load.
OPENPI_CHECKPOINTS: dict[str, str] = {
    "pi0_aloha_pen_uncap": "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap",
}
# SGLang lives in its own conda env so its torch/CUDA stack does not fight the
# main fastkernels env. The setup script is the source of truth for create +
# install; provision just shells out to it. bench_sglang defaults
# --sglang-python to this path.
SGLANG_ENV_NAME = os.environ.get("FASTKERNELS_SGLANG_ENV", "sglang-bench")
_REPO_ROOT = Path(__file__).resolve().parents[2]
SGLANG_SETUP_SCRIPT = _REPO_ROOT / "tests" / "setup_sglang_env.sh"


def _sglang_python() -> Path:
    """``.../envs/sglang-bench/bin/python``, resolved via ``conda info --base``."""
    conda = shutil.which("conda")
    if conda is None:
        # Fall back to the conventional miniconda layout next to the active env.
        base = Path(sys.prefix).resolve().parents[1]  # .../envs/dev -> .../miniconda3
    else:
        base = Path(
            subprocess.run(
                [conda, "info", "--base"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    return base / "envs" / SGLANG_ENV_NAME / "bin" / "python"


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


def _prov_openpi() -> None:
    """Clone OpenPI, build its venv, and convert its Pi0 checkpoints to PyTorch.

    Three things have to exist before bench_openpi can run a reference:
      1. the repo (its `examples/convert_jax_model_to_pytorch.py` is the only
         supported way to produce PyTorch Pi0 weights),
      2. its own venv -- openpi pins jax[cuda12]==0.5.3 and torch==2.7.1, which
         cannot share an environment with the fastkernels torch,
      3. a *converted* checkpoint: the published trees are JAX/orbax, and
         `--openpi-backend pytorch` (the default) requires model.safetensors.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "uv is required to build the OpenPI venv (openpi ships a uv.lock). "
            "Install it from https://docs.astral.sh/uv/ and re-run."
        )
    _git_clone("https://github.com/Physical-Intelligence/openpi", OPENPI_DIR)
    if OPENPI_VENV_PYTHON.is_file():
        print(f"    [skip] OpenPI venv already built: {OPENPI_VENV_PYTHON}", flush=True)
    else:
        _run([uv, "sync", "--frozen"], cwd=OPENPI_DIR)
    _align_openpi_torch_to_gpu(uv)
    _install_openpi_transformers_overlay()

    for name, jax_uri in OPENPI_CHECKPOINTS.items():
        out_dir = OPENPI_ASSETS_DIR / f"{name}_pytorch"
        if _openpi_checkpoint_ready(out_dir):
            print(f"    [skip] already converted: {out_dir}", flush=True)
            continue
        print(f"    downloading OpenPI checkpoint {jax_uri}", flush=True)
        jax_dir = subprocess.run(
            [
                str(OPENPI_VENV_PYTHON), "-c",
                "from openpi.shared import download;"
                f"print(download.maybe_download({jax_uri!r}))",
            ],
            cwd=str(OPENPI_DIR),
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().splitlines()[-1]
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(OPENPI_VENV_PYTHON),
                str(OPENPI_DIR / "examples" / "convert_jax_model_to_pytorch.py"),
                "--checkpoint_dir", jax_dir,
                "--config_name", name,
                "--output_path", str(out_dir),
            ],
            cwd=OPENPI_DIR,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        # The upstream script looks for `assets/` one level *above* the checkpoint
        # dir, so a downloaded tree (which nests assets/ inside it) silently loses
        # its norm_stats.json -- and both engines then skip state/action
        # normalization instead of failing. Copy it here.
        assets_src = Path(jax_dir) / "assets"
        assets_dst = out_dir / "assets"
        if assets_src.is_dir() and not assets_dst.exists():
            print(f"    copying norm stats: {assets_src} -> {assets_dst}", flush=True)
            shutil.copytree(assets_src, assets_dst)
        if not _openpi_checkpoint_ready(out_dir):
            raise RuntimeError(
                f"OpenPI checkpoint conversion did not produce a usable tree at {out_dir} "
                f"(need model.safetensors + assets/**/norm_stats.json)"
            )


def _openpi_venv_supports_local_gpu() -> bool:
    cc = _torch_cc()
    return subprocess.run(
        [
            str(OPENPI_VENV_PYTHON), "-c",
            f"import sys, torch; sys.exit(0 if 'sm_{cc[0]}{cc[1]}' in torch.cuda.get_arch_list() else 1)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(OPENPI_DIR),
    ).returncode == 0


def _align_openpi_torch_to_gpu(uv: str) -> None:
    """Replace openpi's torch wheel when it has no kernels for the local GPU.

    openpi's lock pins ``torch==2.7.1``, and the default PyPI wheel for that
    version is a CUDA 12.6 build whose arch list stops at ``sm_90``. On Blackwell
    (``sm_100``) every launch in the reference then dies with "no kernel image is
    available for execution on the device". The cu128 build of the *same* version
    carries sm_100, so the pin is preserved and only the CUDA flavor changes.
    """
    if _openpi_venv_supports_local_gpu():
        print("    [skip] OpenPI venv torch already targets this GPU", flush=True)
        return
    cc = _torch_cc()
    print(
        f"    OpenPI venv torch has no sm_{cc[0]}{cc[1]} kernels; installing the cu128 build",
        flush=True,
    )
    _run(
        [
            uv, "pip", "install",
            "--python", str(OPENPI_VENV_PYTHON),
            "--index-url", "https://download.pytorch.org/whl/cu128",
            "torch==2.7.1+cu128", "torchvision==0.22.1+cu128",
        ],
        cwd=OPENPI_DIR,
    )
    if not _openpi_venv_supports_local_gpu():
        raise RuntimeError(
            f"OpenPI venv torch still has no sm_{cc[0]}{cc[1]} kernels after the cu128 "
            f"install; pick a wheel matching this GPU by hand in {OPENPI_DIR}/.venv"
        )


def _install_openpi_transformers_overlay() -> None:
    """Copy openpi's `transformers_replace` overlay into its venv's transformers.

    openpi's PyTorch Pi0 refuses to build without it (`transformers_replace is not
    installed correctly`), and `uv sync` does not apply it -- upstream documents it
    as a manual post-install step. Idempotent, and re-run after every sync since a
    re-sync restores the stock transformers.
    """
    overlay = OPENPI_DIR / "src" / "openpi" / "models_pytorch" / "transformers_replace"
    if not overlay.is_dir():
        raise RuntimeError(f"OpenPI transformers overlay missing: {overlay}")
    site = subprocess.run(
        [
            str(OPENPI_VENV_PYTHON), "-c",
            "import os, transformers; print(os.path.dirname(transformers.__file__))",
        ],
        cwd=str(OPENPI_DIR),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()[-1]
    print(f"    applying transformers overlay -> {site}", flush=True)
    shutil.copytree(overlay, site, dirs_exist_ok=True)


def _openpi_checkpoint_ready(out_dir: Path) -> bool:
    return (out_dir / "model.safetensors").is_file() and bool(
        list(out_dir.glob("assets/**/norm_stats.json"))
    )


def _check_openpi() -> bool:
    if not OPENPI_VENV_PYTHON.is_file():
        return False
    importable = subprocess.run(
        [str(OPENPI_VENV_PYTHON), "-c", "import openpi.training.config"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(OPENPI_DIR),
    ).returncode == 0
    return importable and all(
        _openpi_checkpoint_ready(OPENPI_ASSETS_DIR / f"{name}_pytorch")
        for name in OPENPI_CHECKPOINTS
    )


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


def _prov_bitnet() -> None:
    """Microsoft BitNet GPU reference: kernel build + int2/fp16 checkpoints.

    Mirrors the upstream one-time setup in BitNet/gpu/README.md. Two deviations,
    both deliberate:

    * The kernel is built for the *local* compute capability rather than
      upstream's hardcoded ``compute_80`` PTX, matching how instant-ngp/tcnn is
      built here. Otherwise the reference would run JIT-ed sm_80 PTX on a
      Blackwell GPU, which is not a fair reference measurement.
    * ``xformers`` is installed because the reference's ``model.py`` imports
      ``RMSNorm``/``fmha``/``rope_padded`` from it. Current xformers only
      requires ``torch>=2.10``, so it does not perturb the env's torch.
    """
    _git_clone("https://github.com/microsoft/BitNet", BITNET_DIR)
    if not _importable("xformers"):
        _pip("xformers")

    if not BITNET_KERNEL_SO.is_file():
        cc = _torch_cc()
        arch = f"{cc[0]}{cc[1]}"
        _run(
            [
                "nvcc", "-std=c++17", "-Xcudafe", "--diag_suppress=177",
                "--compiler-options", "-fPIC", "-lineinfo", "--shared",
                "bitnet_kernels.cu", "-lcuda",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
                "-o", "libbitnet.so",
            ],
            cwd=BITNET_KERNEL_SO.parent,
        )

    int2 = BITNET_CKPT_DIR / "model_state_int2.pt"
    fp16 = BITNET_CKPT_DIR / "model_state_fp16.pt"
    if int2.is_file() and fp16.is_file():
        print(f"    [skip] already present: {int2.name}, {fp16.name}", flush=True)
        return

    bf16_dir = BITNET_CKPT_DIR / "bitnet-b1.58-2B-4T-bf16"
    safetensors = bf16_dir / "model.safetensors"
    if not safetensors.is_file():
        BITNET_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        _run([
            "hf", "download", "microsoft/bitnet-b1.58-2B-4T-bf16",
            "--local-dir", str(bf16_dir),
        ])

    # convert_safetensors.py rewrites the published bf16 safetensors into the
    # reference's own key layout (same dtype, ~4.8 GB); convert_checkpoint.py
    # then splits that into the int2 and fp16 halves the official CUDA-graph
    # path loads. The intermediate is not needed afterwards -- upstream's README
    # deletes it too.
    intermediate = BITNET_CKPT_DIR / "model_state.pt"
    _run(
        [
            sys.executable, "convert_safetensors.py",
            "--safetensors_file", str(safetensors),
            "--output", str(intermediate),
            "--model_name", "2B",
        ],
        cwd=BITNET_GPU_DIR,
    )
    _run(
        [sys.executable, "convert_checkpoint.py", "--input", str(intermediate)],
        cwd=BITNET_GPU_DIR,
    )
    intermediate.unlink(missing_ok=True)


def _check_bitnet() -> bool:
    return (
        BITNET_KERNEL_SO.is_file()
        and (BITNET_CKPT_DIR / "model_state_int2.pt").is_file()
        and (BITNET_CKPT_DIR / "model_state_fp16.pt").is_file()
        and _importable("xformers")
    )


def _prov_sglang() -> None:
    """Create the isolated ``sglang-bench`` conda env used by ``bench_sglang``.

    SGLang's preferred torch/CUDA cannot share the main fastkernels env, so the
    harness launches the reference side under a separate interpreter. The
    ``tests/setup_sglang_env.sh`` script owns the create + install details;
    provision just runs it (idempotent: it reuses the env if it already exists
    and re-installs ``sglang``).
    """
    if not SGLANG_SETUP_SCRIPT.is_file():
        raise RuntimeError(
            f"SGLang setup script missing: {SGLANG_SETUP_SCRIPT}. "
            "Expected tests/setup_sglang_env.sh next to the package."
        )
    if shutil.which("conda") is None:
        raise RuntimeError(
            "conda is required to build the sglang-bench env "
            "(SGLang cannot share the fastkernels torch). Install Miniconda/Anaconda "
            "and re-run, or create the env by hand: bash tests/setup_sglang_env.sh"
        )
    env = {**os.environ, "ENV_NAME": SGLANG_ENV_NAME}
    _run(["bash", str(SGLANG_SETUP_SCRIPT)], env=env)


def _check_sglang() -> bool:
    py = _sglang_python()
    if not py.is_file():
        return False
    return (
        subprocess.run(
            [str(py), "-c", "import sglang, torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


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
        Component("openpi", "OpenPI reference venv + converted PyTorch Pi0 checkpoint", ("bench_openpi",), _prov_openpi, _check_openpi),
        Component("bitnet", "Microsoft BitNet GPU kernel + int2/fp16 checkpoints", ("bench_microsoft_bitnet",), _prov_bitnet, _check_bitnet),
        Component(
            "sglang",
            "isolated sglang-bench conda env (SGLang EAGLE-3 reference)",
            ("bench_sglang",),
            _prov_sglang,
            _check_sglang,
        ),
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
