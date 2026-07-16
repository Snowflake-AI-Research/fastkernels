"""``fastkernels validate`` — run the proper reference-library harness per model.

Resolves a scenario table (a path, or a packaged name like ``full`` / ``default``
/ ``minimal`` under ``fastkernels/scenarios/``) and, for each scenario, runs the
``bench_*.py`` harness that validates that model's family against its SOTA
reference library (vLLM, SGLang, FLA, diffusers, timm, …). The model→harness
choice mirrors ``capture.py``'s per-model dispatch: name-pattern predicates first,
then ``workloads.module_for`` → an explicit module→harness table.

A single ``--max-requests`` / ``--max-layers`` is translated to each harness's own
flags (harnesses that lack an analog ignore it, with a note). Scenarios are packed
across the available GPUs by their TP degree; each harness runs as its own
subprocess (harnesses are script-shaped and already isolate their own process).

Usage::

    fastkernels validate minimal                       # all-GPU, full workload
    fastkernels validate full --max-requests 8 --max-layers 12
    fastkernels validate minimal --dry-run             # print the plan, run nothing
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

_VALIDATE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VALIDATE_DIR.parent.parent

# --- model -> harness dispatch (mirrors capture.py's per-model predicates) ---
# module stem (from workloads.module_for) -> harness filename (no .py).
_MODULE_TO_HARNESS: dict[str, str] = {
    # LLMs / VLMs / ASR — bench_vllm routes VLM/omni/whisper internally by --model.
    "llama": "bench_vllm", "deepseek": "bench_vllm", "mixtral": "bench_vllm",
    "gpt_oss": "bench_vllm", "gemma4": "bench_vllm", "mamba": "bench_vllm",
    "mamba2": "bench_vllm", "qwen2_vl": "bench_vllm", "qwen3_vl": "bench_vllm",
    "qwen2_5_omni": "bench_vllm", "whisper": "bench_vllm",
    # diffusion / video / TTS
    "flux": "bench_vllm_omni", "hunyuan_video": "bench_vllm_omni",
    "cosyvoice3": "bench_vllm_omni", "sdxl": "bench_diffusers",
    # vision encoders / classification / detection / segmentation
    "sam3": "bench_sam", "siglip2": "bench_timm", "dinov3": "bench_timm",
    "swinv2": "bench_timm", "mobilenetv4": "bench_timm",
    "convnextv2": "bench_image_cls", "efficientnetv2": "bench_image_cls",
    "yolov10": "bench_detection", "rtdetrv2": "bench_detection",
    # embeddings / recsys
    "bge_m3": "bench_embedding", "colbertv2": "bench_embedding",
    "dlrmv2": "bench_recsys", "lightgcn": "bench_recsys",
    # 3D / robotics / science / world models
    "gaussian_splatting": "bench_3dgs", "instant_ngp": "bench_instantngp",
    "pointtransformerv3": "bench_pointcloud", "openfold3": "bench_openfold3",
    "pi0": "bench_openpi", "dp3": "bench_dp3", "oasis": "bench_oasis",
    "vjepa2": "bench_vjepa2", "ttt_e2e": "bench_ttt_e2e", "llada": "bench_dllm",
}


def _harness_for(hf_name: str) -> str | None:
    """Harness filename (no ``.py``) for a model, or ``None`` if unmapped."""
    n = hf_name.lower()
    if "eagle3" in n:
        return "bench_sglang"
    if n.startswith("fla-hub/"):
        return "bench_fla"
    if "jamba" in n:
        return "bench_jamba"
    if "bitnet" in n:
        return "bench_microsoft_bitnet"
    if "llada" in n:
        return "bench_dllm"
    if "stable-diffusion" in n or "sdxl" in n:
        return "bench_diffusers"
    from fastkernels.workloads import module_for
    module = module_for(hf_name)
    return _MODULE_TO_HARNESS.get(module) if module else None


# --- per-harness flag adapter ------------------------------------------------
# How each harness names "number of requests/items". Harnesses not listed take no
# request-count flag, so --max-requests is ignored for them (with a printed note).
_REQUESTS_FLAG = {
    "bench_vllm": "--num-seqs", "bench_fla": "--num-seqs", "bench_jamba": "--num-seqs",
    "bench_sglang": "--num-seqs", "bench_detection": "--num-images",
    "bench_timm": "--num-images",
}
_TP_OK = {"bench_vllm", "bench_embedding"}          # accept --tp
_MAXLAYERS_OK = {"bench_vllm"}                       # accept --max-layers
_EAGER_OK = {"bench_vllm", "bench_sglang", "bench_diffusers", "bench_embedding"}


def _build_cmd(scenario, harness: str, args) -> list[str]:
    cmd = [sys.executable, "-u", str(_VALIDATE_DIR / f"{harness}.py"),
           "--model", scenario.hf_name]
    if harness in _TP_OK and scenario.tp:
        cmd += ["--tp", str(scenario.tp)]
    if args.max_layers is not None and harness in _MAXLAYERS_OK:
        cmd += ["--max-layers", str(args.max_layers)]
    if args.max_requests is not None:
        flag = _REQUESTS_FLAG.get(harness)
        if flag:
            cmd += [flag, str(args.max_requests)]
        else:
            print(f"    note: {harness} has no request-count flag; "
                  f"--max-requests ignored")
    if scenario.enforce_eager and harness in _EAGER_OK:
        cmd += ["--enforce-eager"]
    return cmd


# --- GPU pool ----------------------------------------------------------------
def _detect_gpus(explicit: str | None) -> list[str]:
    if explicit:
        return [t.strip() for t in explicit.split(",") if t.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [t.strip() for t in env.split(",") if t.strip()]
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True).stdout
            ids = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if ids:
                return ids
        except Exception:  # noqa: BLE001
            pass
    return ["0"]


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        for _ in range(40):
            if proc.poll() is not None:
                return
            time.sleep(0.2)


# --- output styling ----------------------------------------------------------
def _c(text: str, code: str) -> str:
    """ANSI-wrap when stdout is a TTY, else return plain."""
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def _tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


# --- scheduler ---------------------------------------------------------------
def _run(scenarios, args, gpus: list[str]) -> int:
    timeout = int(os.environ.get("FASTKERNELS_VALIDATE_TIMEOUT_SEC", "3600"))
    log_dir = (Path(args.output_dir) if args.output_dir
               else _REPO_ROOT / "tests" / "results" / "validate_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []  # (index, scenario, harness)
    results: dict[int, str] = {}
    for i, s in enumerate(scenarios):
        harness = _harness_for(s.hf_name)
        if harness is None:
            print(f"  {_c('−', '2')} skip {s.hf_name}: no harness mapped for this model")
            results[i] = "SKIP(no-harness)"
        elif s.tp > len(gpus):
            print(f"  {_c('−', '2')} skip {s.hf_name}: needs tp={s.tp} > {len(gpus)} GPU(s)")
            results[i] = "SKIP(tp>gpus)"
        else:
            jobs.append((i, s, harness))

    def _env(held):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(held)
        env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        return env

    def _summary() -> int:
        print(_c("\nvalidate summary", "1"))
        for i, s in enumerate(scenarios):
            r = results.get(i, "?")
            mark = (_c("✓", "32") if r == "PASS"
                    else _c("−", "2") if r.startswith("SKIP") else _c("✗", "31"))
            print(f"  {mark} {r:16} {s.hf_name}")
        return 0 if all(v == "PASS" for v in results.values()) else 1

    # Single scenario: stream the harness straight to the terminal so it looks
    # exactly like running the bench_*.py directly (its own tqdm bars and all).
    if len(jobs) == 1:
        i, s, harness = jobs[0]
        print(f"\n{_c('▶', '36')} {_c(s.hf_name, '1')} {_c('→ ' + harness, '2')}"
              f"  (tp={s.tp}, dtype={s.dtype})\n", flush=True)
        proc = subprocess.Popen(_build_cmd(s, harness, args), env=_env(gpus[: s.tp]),
                                cwd=str(_REPO_ROOT), start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(_c(f"\ntimeout after {timeout}s — killing {harness}", "31"))
            _kill_group(proc)
            rc = proc.poll() if proc.poll() is not None else -9
        results[i] = "PASS" if rc == 0 else f"FAIL(rc={rc})"
        mark = _c("✓ PASS", "32") if rc == 0 else _c(f"✗ FAIL (rc={rc})", "31")
        print(f"\n{mark}  {s.hf_name}  ·  {harness}", flush=True)
        return 0 if rc == 0 else 1

    if not jobs:
        return _summary()

    # Multiple scenarios: per-scenario logfiles (can't interleave live) + a halo
    # progress spinner. Clean start/finish lines; failures show a short log tail.
    print(f"\n{_c('▶', '36')} validating {_c(str(len(jobs)), '1')} scenario(s) across "
          f"{len(gpus)} GPU(s)  ·  logs in {log_dir}\n", flush=True)
    try:
        from halo import Halo
        spinner = Halo(spinner="dots", stream=sys.stderr)
    except Exception:  # noqa: BLE001 - spinner is cosmetic
        spinner = None

    pending = deque(sorted(jobs, key=lambda j: j[1].tp, reverse=True))
    free = list(gpus)
    running: dict[subprocess.Popen, dict] = {}

    def _spin_text() -> str:
        parts = [f"[{d['i']}] {d['name'].split('/')[-1]} {int(time.monotonic() - d['start'])}s"
                 for d in running.values()]
        return "  running " + " | ".join(parts) if parts else "  scheduling…"

    def _emit(line: str) -> None:
        if spinner:
            spinner.stop()
        print(line, flush=True)
        if spinner and (pending or running):
            spinner.start(_spin_text())

    def _launch(i, s, harness):
        held = [free.pop(0) for _ in range(s.tp)]
        log_path = log_dir / f"{i}_{harness}_{s.hf_name.replace('/', '__')}.log"
        lf = open(log_path, "w")
        proc = subprocess.Popen(_build_cmd(s, harness, args), stdout=lf,
                                stderr=subprocess.STDOUT, env=_env(held),
                                cwd=str(_REPO_ROOT), start_new_session=True)
        running[proc] = {"i": i, "held": held, "lf": lf, "log": log_path,
                         "start": time.monotonic(), "harness": harness, "name": s.hf_name}
        _emit(f"{_c('▷', '36')} [{i}] {s.hf_name} {_c('→ ' + harness, '2')}  "
              f"tp={s.tp} GPU={','.join(held)}")

    if spinner:
        spinner.start(_spin_text())
    while pending or running:
        while pending and len(free) >= pending[0][1].tp:
            _launch(*pending.popleft())
        if running:
            time.sleep(1.0)
            if spinner:
                spinner.text = _spin_text()
        for proc in list(running):
            info = running[proc]
            done = proc.poll() is not None
            if not done and time.monotonic() - info["start"] > timeout:
                _kill_group(proc)
                done = True
            if done:
                info["lf"].close()
                rc = proc.returncode
                ok = rc == 0
                results[info["i"]] = "PASS" if ok else f"FAIL(rc={rc})"
                el = int(time.monotonic() - info["start"])
                mark = _c("✓", "32") if ok else _c("✗", "31")
                status = "PASS" if ok else f"FAIL rc={rc}"
                free.extend(info["held"])
                del running[proc]
                _emit(f"{mark} [{info['i']}] {info['name']}  ·  {info['harness']}  ·  "
                      f"{status} ({el}s)  ·  {info['log']}")
                if not ok:
                    for ln in _tail(info["log"], 12).splitlines():
                        print(f"    {_c(ln, '2')}")
    if spinner:
        spinner.stop()
    return _summary()



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="fastkernels validate",
        description="Run the proper reference-library bench harness for each model "
                    "in a scenario table (fastkernels vs SOTA reference).")
    p.add_argument("scenarios",
                   help="Scenario table: a path, or a packaged name (full/default/"
                        "minimal) resolved against fastkernels/scenarios/.")
    p.add_argument("--max-requests", type=int, default=None,
                   help="Cap requests/items per harness (translated to its own flag, "
                        "e.g. --num-seqs / --num-images; ignored where unsupported).")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Run only the first N decoder layers (LLM harnesses only).")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids to pack across (default: all visible).")
    p.add_argument("--output-dir", default=None,
                   help="Directory for per-scenario dispatcher logs "
                        "(default: tests/results/validate_logs).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the chosen harness + command per scenario; run nothing.")
    args = p.parse_args(argv)

    from fastkernels.workloads import resolve_benchmark
    try:
        scenarios = resolve_benchmark(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: could not load scenarios {args.scenarios!r}: {exc}")
        return 2

    if args.dry_run:
        for i, s in enumerate(scenarios):
            harness = _harness_for(s.hf_name)
            if harness is None:
                print(f"[{i}] {s.hf_name}: NO HARNESS MAPPED")
                continue
            cmd = _build_cmd(s, harness, args)
            print(f"[{i}] {s.hf_name}  (tp={s.tp}, dtype={s.dtype}) -> {harness}")
            print("      " + " ".join(cmd))
        return 0

    return _run(scenarios, args, _detect_gpus(args.gpus))


if __name__ == "__main__":
    raise SystemExit(main())
