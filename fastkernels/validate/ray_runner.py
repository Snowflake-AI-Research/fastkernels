"""Ray execution support for :mod:`fastkernels.validate`."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from math import floor
from pathlib import Path

from . import (
    _REPO_ROOT,
    _build_cmd,
    _harness_for,
    _safe_slug,
    _scenario_workloads,
)
from .. import CACHE_DIR

_RAY_PROGRESS_INTERVAL_SEC = 30.0
_TASK_RESULT_FILE = "task_result.json"


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def _tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _job_paths(
    root: Path,
    index: int,
    scenario,
    harness: str,
) -> tuple[Path, Path]:
    name = _safe_slug(f"{index:03d}_{harness}_{scenario.hf_name}")
    run_dir = root / name
    return run_dir, run_dir / "run.log"


# Compiler-cache subdirectories, one env var each. Keys are the env var names
# the toolchains read; values are the subdirectory under this run's cache root.
_CACHE_SUBDIRS = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    # One root covering vLLM's torch_compile_cache, flashinfer_autotune_cache,
    # deep_gemm and modelinfos caches.
    "VLLM_CACHE_ROOT": "vllm",
    "CUDA_CACHE_PATH": "cuda",
}


def _run_cache_root(root: Path) -> Path:
    """Cache root for this validate run.

    Keyed on the run id (``root.name``), so a fresh run starts with empty
    compiler caches and ``--resume`` -- which resolves back to the same run id
    -- reuses whatever the interrupted run already compiled. Keeping the caches
    per-run means one run's accumulated warmth cannot silently change the next
    run's timings; clearing them all is ``rm -rf`` on one directory.
    """
    return CACHE_DIR / root.name


def _cache_env(cache_root: Path) -> dict[str, str]:
    """Env pointing every compiler cache at ``cache_root``, creating the dirs.

    Assigned rather than defaulted: the point is a known cache state per run,
    so a stray ``TRITON_CACHE_DIR`` in the ambient environment must not quietly
    reintroduce cross-run sharing. Relocate the whole tree with
    ``FASTKERNELS_CACHE_DIR`` instead.
    """
    env = {}
    for var, subdir in _CACHE_SUBDIRS.items():
        path = cache_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[var] = str(path)
    return env


def _make_job(index: int, scenario, harness: str, args, root: Path) -> dict:
    run_dir, log_path = _job_paths(root, index, scenario, harness)
    return {
        "index": index,
        "name": scenario.hf_name,
        "draft_model": getattr(scenario, "draft_model", None),
        "tp": int(scenario.tp),
        "dtype": scenario.dtype,
        "harness": harness,
        "workloads": list(_scenario_workloads(scenario)),
        "cmd": _build_cmd(scenario, harness, args, run_dir),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "cache_root": str(_run_cache_root(root)),
    }


def _task_result_path(job: dict) -> Path:
    return Path(job["run_dir"]) / _TASK_RESULT_FILE


def _load_cached_result(job: dict) -> dict | None:
    marker_path = _task_result_path(job)
    if marker_path.is_file():
        try:
            result = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError):
            result = None
        if (
            isinstance(result, dict)
            and str(result.get("status", "")).startswith("PASS")
            and result.get("cmd", job["cmd"]) == job["cmd"]
        ):
            return {
                **job,
                **result,
                "returncode": 0,
                "elapsed_s": 0.0,
                "status": "PASS(cached)",
                "cached": True,
            }
        # A marker that exists but does not record a PASS for this exact command
        # is authoritative: the scenario ran and failed (or ran a different
        # command), so --resume must run it again. Falling through to the legacy
        # artifact check here would resurrect it as PASS(cached) purely because a
        # partial results file survived the failure.
        return None

    # Compatibility with runs created before task_result.json was introduced.
    run_dir = Path(job["run_dir"])
    legacy_artifact = (
        run_dir / "summary.json"
        if job["harness"] == "bench_openpi"
        else run_dir / "results.json"
    )
    if legacy_artifact.is_file():
        return {
            **job,
            "returncode": 0,
            "elapsed_s": 0.0,
            "status": "PASS(cached)",
            "cached": True,
        }
    return None


def _plan_jobs(
    scenarios,
    args,
    total_gpus: int,
    root: Path,
) -> tuple[list[dict], dict[int, str], list[dict]]:
    jobs: list[dict] = []
    cached: list[dict] = []
    results: dict[int, str] = {}
    for index, scenario in enumerate(scenarios):
        harness = _harness_for(
            scenario.hf_name, getattr(scenario, "draft_model", None)
        )
        if harness is None:
            print(f"  - skip {scenario.hf_name}: no harness mapped")
            results[index] = "SKIP(no-harness)"
            continue
        if scenario.tp > total_gpus:
            print(
                f"  - skip {scenario.hf_name}: needs tp={scenario.tp} "
                f"> {total_gpus} GPU(s)"
            )
            results[index] = "SKIP(tp>gpus)"
            continue
        job = _make_job(index, scenario, harness, args, root)
        cached_result = _load_cached_result(job) if args.resume else None
        if cached_result is not None:
            print(
                f"  {_c('cached', '32')} {scenario.hf_name}: "
                f"{job['run_dir']}"
            )
            results[index] = "PASS(cached)"
            cached.append(cached_result)
        else:
            jobs.append(job)
    return jobs, results, cached


def _visible_to_physical_gpu_ids(
    gpu_ids: list[str],
    parent_visible: list[str],
) -> list[str]:
    out: list[str] = []
    for gpu_id in gpu_ids:
        if gpu_id in parent_visible:
            out.append(gpu_id)
            continue
        try:
            index = int(gpu_id)
        except ValueError:
            out.append(gpu_id)
            continue
        if parent_visible and 0 <= index < len(parent_visible):
            out.append(parent_visible[index])
        else:
            out.append(gpu_id)
    return out


def _numa_nodes_for_gpus(gpu_ids: list[str]) -> list[str]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    nodes: set[str] = set()
    for gpu_id in gpu_ids:
        try:
            output = subprocess.run(
                [smi, "topo", "-C", "-i", gpu_id],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout
        except Exception:
            continue
        match = re.search(r"NUMA IDs of closest CPU:\s*([0-9, -]+)", output)
        if match:
            nodes.update(
                token
                for token in re.split(r"[, ]+", match.group(1).strip())
                if token
            )
    return sorted(nodes, key=int)


def _numactl_prefix(gpu_ids: list[str], mode: str) -> list[str]:
    if mode == "off":
        return []
    numactl = shutil.which("numactl")
    if numactl is None:
        return []
    nodes = _numa_nodes_for_gpus(gpu_ids)
    if not nodes:
        return []
    joined = ",".join(nodes)
    prefix = [numactl, f"--cpunodebind={joined}"]
    if mode == "strict":
        prefix.append(f"--membind={joined}")
    return prefix


def _total_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0


def _reserved_cpus(total_cpus: int) -> int:
    return min(max(4, total_cpus // 16), max(1, total_cpus - 1))


def _reserved_memory_bytes(total_memory: int) -> int:
    if total_memory <= 0:
        return 0
    sixteen_gib = 16 * 1024**3
    return min(
        max(sixteen_gib, total_memory // 10),
        max(0, total_memory - 1024**3),
    )


def _ray_resource_options(
    tp: int,
    total_gpus: int,
    cluster_resources: dict,
) -> dict:
    total_cpus = int(cluster_resources.get("CPU") or os.cpu_count() or 1)
    alloc_cpus = max(1, total_cpus - _reserved_cpus(total_cpus))
    fraction = tp / max(1, total_gpus)
    options = {
        "num_gpus": tp,
        "num_cpus": max(1, floor(alloc_cpus * fraction)),
    }
    total_memory = int(cluster_resources.get("memory") or _total_memory_bytes())
    alloc_memory = max(0, total_memory - _reserved_memory_bytes(total_memory))
    if alloc_memory:
        options["memory"] = max(
            256 * 1024**2,
            int(alloc_memory * fraction * 0.85),
        )
    return options


def _reclaim_gpus(gpu_ids: list[str]) -> None:
    smi = shutil.which("nvidia-smi")
    selected_ids = [gpu_id for gpu_id in gpu_ids if gpu_id]
    if smi is None or not selected_ids:
        return
    try:
        output = subprocess.run(
            [
                smi,
                "-i",
                ",".join(selected_ids),
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return
    for token in output.split():
        try:
            os.kill(int(token), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


def _kill_process_group(
    proc: subprocess.Popen,
    physical_gpus: list[str],
) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        for _ in range(40):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is not None:
            break
    time.sleep(1.0)
    _reclaim_gpus(physical_gpus)


def _run_job_subprocess(
    job: dict,
    timeout: int,
    stall_timeout: int,
    repo_root: str,
    parent_visible_gpus: list[str],
    numactl_mode: str,
) -> dict:
    start = time.monotonic()
    run_dir = Path(job["run_dir"])
    log_path = Path(job["log_path"])
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["FASTKERNELS_VALIDATE_JOB_INDEX"] = str(job["index"])
    env["FASTKERNELS_VALIDATE_JOB_NAME"] = _safe_slug(job["name"], max_len=120)
    cache_root = Path(job["cache_root"])
    env.update(_cache_env(cache_root))
    visible_gpus = [
        token.strip()
        for token in env.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    physical_gpus = _visible_to_physical_gpu_ids(
        visible_gpus,
        parent_visible_gpus,
    )
    prefix = _numactl_prefix(physical_gpus or visible_gpus, numactl_mode)
    cmd = [*prefix, *job["cmd"]]

    proc: subprocess.Popen | None = None
    watchdog_reason: str | None = None
    last_output = [time.monotonic()]
    with log_path.open("w", buffering=1) as log:
        log.write(f"job_index: {job['index']}\n")
        log.write(f"name: {job['name']}\n")
        if job.get("draft_model"):
            log.write(f"draft_model: {job['draft_model']}\n")
        log.write(f"harness: {job['harness']}\n")
        log.write(f"tp: {job['tp']}\n")
        log.write(f"workloads: {', '.join(job['workloads'])}\n")
        log.write(f"CUDA_VISIBLE_DEVICES: {','.join(visible_gpus)}\n")
        log.write(f"physical_gpus: {','.join(physical_gpus)}\n")
        log.write(f"numactl_prefix: {' '.join(prefix)}\n")
        log.write(f"cache_root: {cache_root}\n")
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=repo_root,
                start_new_session=True,
            )

            def _pump_stdout() -> None:
                assert proc is not None
                if proc.stdout is None:
                    return
                for line in proc.stdout:
                    last_output[0] = time.monotonic()
                    log.write(line)
                    # Echo to the worker's own stdout so the child's output is
                    # captured in Ray's per-task worker log and viewable in the
                    # dashboard (the log file above remains the source of truth).
                    # The worker stdout is block-buffered when redirected to a
                    # file, so flush per line to keep the dashboard view live.
                    sys.stdout.write(line)
                    sys.stdout.flush()

            pump = threading.Thread(target=_pump_stdout, daemon=True)
            pump.start()
            while proc.poll() is None:
                now = time.monotonic()
                if timeout > 0 and now - start > timeout:
                    watchdog_reason = f"timeout after {timeout}s"
                    break
                if stall_timeout > 0 and now - last_output[0] > stall_timeout:
                    watchdog_reason = f"no log output for {stall_timeout}s"
                    break
                time.sleep(1.0)
            if watchdog_reason:
                log.write(f"\nWATCHDOG: {watchdog_reason}\n")
                log.flush()
                _kill_process_group(proc, physical_gpus)
            pump.join(timeout=5.0)
        except BaseException:
            if proc is not None and proc.poll() is None:
                _kill_process_group(proc, physical_gpus)
            raise

    elapsed = time.monotonic() - start
    returncode = proc.returncode if proc is not None and proc.returncode is not None else -9
    if watchdog_reason:
        status = f"FAIL({watchdog_reason})"
    else:
        status = "PASS" if returncode == 0 else f"FAIL(rc={returncode})"
    result = {
        "index": job["index"],
        "name": job["name"],
        "harness": job["harness"],
        "tp": job["tp"],
        "dtype": job["dtype"],
        "workloads": job["workloads"],
        "cmd": job["cmd"],
        "returncode": returncode,
        "watchdog_reason": watchdog_reason,
        "elapsed_s": elapsed,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "visible_gpus": visible_gpus,
        "physical_gpus": physical_gpus,
        "status": status,
    }
    if status == "PASS":
        marker_path = _task_result_path(job)
        temporary_path = marker_path.with_suffix(".tmp")
        try:
            temporary_path.write_text(json.dumps(result, indent=2) + "\n")
            temporary_path.replace(marker_path)
        except OSError:
            temporary_path.unlink(missing_ok=True)
    return result


def _ray_run_job(
    job: dict,
    timeout: int,
    stall_timeout: int,
    repo_root: str,
    parent_visible_gpus: list[str],
    numactl_mode: str,
) -> dict:
    return _run_job_subprocess(
        job,
        timeout,
        stall_timeout,
        repo_root,
        parent_visible_gpus,
        numactl_mode,
    )


def _append_run_event(root: Path, event: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.jsonl").open("a") as output:
        output.write(json.dumps({"ts": time.time(), **event}, sort_keys=True) + "\n")


def _scenario_label(name_or_path: str | Path) -> str:
    return _safe_slug(Path(str(name_or_path)).stem, max_len=None)


def _ray_job_id(scenario_name: str, job: dict) -> str:
    model = _safe_slug(job["name"], max_len=None)
    workloads = _safe_slug(
        "_".join(job.get("workloads") or ["workloads"]),
        max_len=None,
    )
    return (
        f"validate_{scenario_name}_{job['index']:03d}_{model}_"
        f"tp{job['tp']}_{workloads}"
    )


def _dashboard_url(ray) -> str | None:
    try:
        get_url = getattr(ray, "get_dashboard_url", None)
        value = get_url() if get_url else None
        if value:
            return value if value.startswith("http") else f"http://{value}"
    except Exception:
        pass
    try:
        value = ray._private.worker._global_node.webui_url
        return value if value.startswith("http") else f"http://{value}"
    except Exception:
        return None


def _load_task_results(root: Path) -> dict[int, dict]:
    results: dict[int, dict] = {}
    path = root / "run.jsonl"
    if not path.is_file():
        return results
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "task_finished":
            continue
        result = event.get("result") or {}
        index = result.get("index")
        if isinstance(index, int):
            results[index] = result
    return results


def _fmt_speedup(value) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return f"{value:.2f}x"


def _alignment_summary(alignment: dict | None) -> str:
    if not isinstance(alignment, dict):
        return "-"
    average = alignment.get("avg_matching_tokens_per_request")
    exact = alignment.get("exact_matches")
    total = alignment.get("total_seqs")
    parts: list[str] = []
    if isinstance(average, (int, float)):
        parts.append(f"avg prefix match len: {average:.1f}")
    if isinstance(exact, int) and isinstance(total, int) and total:
        parts.append(f"exact match: {exact}/{total} ({exact / total * 100:.1f}%)")
    return "; ".join(parts) if parts else "-"


def _cosine_summary(items: list[float]) -> str:
    values = [value for value in items if isinstance(value, (int, float))]
    return f"min_cos={min(values):.4f}" if values else "-"


def _reference_name(harness: str) -> str:
    return {
        "bench_vllm": "vLLM",
        "bench_fla": "FLA",
        "bench_jamba": "FLA",
        "bench_sglang": "SGLang",
        "bench_vllm_omni": "vllm-omni",
        "bench_diffusers": "diffusers",
        "bench_timm": "timm",
        "bench_detection": "reference",
        "bench_openfold3": "reference",
        "bench_embedding": "vLLM",
        "bench_oasis": "open-oasis",
    }.get(harness, "reference")


def _throughput_rows_for_result(
    model: str,
    harness: str,
    data: dict,
) -> list[dict]:
    rows: list[dict] = []
    reference = _reference_name(harness)
    if harness in {"bench_vllm", "bench_fla", "bench_jamba", "bench_sglang"}:
        for item in data.get("scenarios", []):
            rows.append(
                {
                    "model": model,
                    "workload": item.get("scenario") or item.get("name"),
                    "reference": reference,
                    "speedup": item.get("speedup"),
                    "correctness": _alignment_summary(item.get("alignment")),
                }
            )
    elif harness == "bench_vllm_omni":
        correctness = data.get("correctness") or {}
        for item in data.get("fastkernels", {}).get("throughput", []):
            name = item.get("scenario") or item.get("name")
            reference_item = next(
                (
                    entry
                    for entry in data.get("vllm_omni", {}).get("throughput", [])
                    if (entry.get("scenario") or entry.get("name")) == name
                ),
                {},
            )
            fastkernels_rate = (
                item.get("images_per_second")
                or item.get("items_per_second")
                or item.get("samples_per_second")
            )
            reference_rate = (
                reference_item.get("images_per_second")
                or reference_item.get("items_per_second")
                or reference_item.get("samples_per_second")
            )
            item_correctness = (
                correctness.get(name, {}) if isinstance(correctness, dict) else {}
            )
            cosines: list[float] = []
            if isinstance(item_correctness, dict):
                for key, value in item_correctness.items():
                    if isinstance(value, dict) and isinstance(
                        value.get("cosine"), (int, float)
                    ):
                        cosines.append(value["cosine"])
                    elif (
                        key.endswith(("cosine", "cosine_sim"))
                        and isinstance(value, (int, float))
                    ):
                        cosines.append(value)
            rows.append(
                {
                    "model": model,
                    "workload": name,
                    "reference": reference,
                    "speedup": (
                        fastkernels_rate / reference_rate
                        if fastkernels_rate and reference_rate
                        else item.get("speedup")
                    ),
                    "correctness": _cosine_summary(cosines),
                }
            )
    elif harness == "bench_detection":
        correctness = data.get("correctness") or {}
        reference_items = data.get("reference", {}).get("throughput", [])
        for index, item in enumerate(
            data.get("fastkernels", {}).get("throughput", [])
        ):
            reference_item = (
                reference_items[index] if index < len(reference_items) else {}
            )
            fastkernels_rate = item.get("images_per_second")
            reference_rate = reference_item.get("images_per_second")
            rows.append(
                {
                    "model": model,
                    "workload": item.get("name"),
                    "reference": reference,
                    "speedup": (
                        fastkernels_rate / reference_rate
                        if fastkernels_rate and reference_rate
                        else item.get("speedup")
                    ),
                    "correctness": (
                        f"boxes={correctness.get('boxes_cosine', 0):.4f}; "
                        f"scores={correctness.get('scores_cosine', 0):.4f}; "
                        "labels="
                        f"{correctness.get('labels_match_rate', 0) * 100:.1f}%"
                    ),
                }
            )
    elif harness == "bench_openfold3":
        for item in data.get("throughput_scenarios", []):
            alignment = item.get("alignment") or {}
            rows.append(
                {
                    "model": model,
                    "workload": item.get("scenario"),
                    "reference": reference,
                    "speedup": item.get("speedup"),
                    "correctness": (
                        f"align={alignment.get('pass_rate', 0) * 100:.1f}%"
                        if alignment
                        else "-"
                    ),
                }
            )
    elif harness == "bench_embedding":
        for item in data.get("throughput_scenarios", []):
            correctness = item.get("correctness") or {}
            rows.append(
                {
                    "model": model,
                    "workload": item.get("scenario"),
                    "reference": reference,
                    "speedup": item.get("speedup"),
                    "correctness": (
                        f"pass={correctness.get('pass')}; "
                        f"min_cos={correctness.get('min_cosine', 0):.6f}"
                        if correctness
                        else "-"
                    ),
                }
            )
    elif harness == "bench_oasis":
        for item in data.get("performance", []):
            scenario = item.get("scenario")
            name = scenario.get("name") if isinstance(scenario, dict) else scenario
            correctness = item.get("correctness") or {}
            cosines: list[float] = []
            passes: list[bool] = []
            if isinstance(correctness, dict):
                for value in correctness.values():
                    if not isinstance(value, dict):
                        continue
                    if isinstance(value.get("pass"), bool):
                        passes.append(value["pass"])
                    if isinstance(value.get("cosine"), (int, float)):
                        cosines.append(value["cosine"])
            summary = _cosine_summary(cosines)
            if passes:
                summary = f"pass={all(passes)}; {summary}"
            rows.append(
                {
                    "model": model,
                    "workload": name,
                    "reference": reference,
                    "speedup": item.get("speedup"),
                    "correctness": summary,
                }
            )
    return rows


def _latency_rows_for_result(
    model: str,
    harness: str,
    data: dict,
) -> list[dict]:
    rows: list[dict] = []
    reference = _reference_name(harness)
    if harness in {
        "bench_vllm",
        "bench_fla",
        "bench_jamba",
        "bench_sglang",
        "bench_embedding",
    }:
        for item in data.get("latency_scenarios", []):
            rows.append(
                {
                    "model": model,
                    "workload": item.get("scenario") or item.get("name"),
                    "reference": reference,
                    "speedup": item.get("speedup"),
                }
            )
    elif harness == "bench_vllm_omni":
        reference_items = data.get("vllm_omni", {}).get("latency", [])
        for index, item in enumerate(
            data.get("fastkernels", {}).get("latency", [])
        ):
            reference_item = (
                reference_items[index] if index < len(reference_items) else {}
            )
            fastkernels_median = item.get("median_s")
            reference_median = reference_item.get("median_s")
            rows.append(
                {
                    "model": model,
                    "workload": item.get("scenario") or item.get("name"),
                    "reference": reference,
                    "speedup": (
                        reference_median / fastkernels_median
                        if fastkernels_median and reference_median
                        else item.get("speedup")
                    ),
                }
            )
    elif harness == "bench_detection":
        reference_items = data.get("reference", {}).get("latency", [])
        for index, item in enumerate(
            data.get("fastkernels", {}).get("latency", [])
        ):
            reference_item = (
                reference_items[index] if index < len(reference_items) else {}
            )
            fastkernels_median = item.get("median_s")
            reference_median = reference_item.get("median_s")
            rows.append(
                {
                    "model": model,
                    "workload": item.get("name"),
                    "reference": reference,
                    "speedup": (
                        reference_median / fastkernels_median
                        if fastkernels_median and reference_median
                        else item.get("speedup")
                    ),
                }
            )
    elif harness == "bench_openfold3":
        for item in data.get("latency_scenarios", []):
            rows.append(
                {
                    "model": model,
                    "workload": item.get("scenario"),
                    "reference": reference,
                    "speedup": item.get("speedup"),
                }
            )
    return rows


def _format_table(rows: list[dict], headers: list[tuple[str, str]]) -> str:
    widths = [len(label) for _key, label in headers]
    rendered: list[list[str]] = []
    for row in rows:
        values: list[str] = []
        for key, _label in headers:
            value = row.get(key)
            if key == "speedup":
                rendered_value = _fmt_speedup(value)
            else:
                rendered_value = "-" if value is None else str(value)
            values.append(rendered_value)
        rendered.append(values)
        for index, value in enumerate(values):
            widths[index] = max(widths[index], len(value))
    lines = [
        "  ".join(
            label.ljust(widths[index])
            for index, (_key, label) in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    for values in rendered:
        lines.append(
            "  ".join(
                values[index].ljust(widths[index])
                for index in range(len(headers))
            )
        )
    return "\n".join(lines)


def _result_artifact_path(run_dir: Path, harness: str | None) -> Path:
    results_path = run_dir / "results.json"
    if results_path.is_file() or harness != "bench_openpi":
        return results_path
    return run_dir / "summary.json"


def _build_summary(root: Path, scenarios, results: dict[int, str]) -> dict:
    task_results = _load_task_results(root)
    throughput: list[dict] = []
    latency: list[dict] = []
    models: list[dict] = []
    for index, scenario in enumerate(scenarios):
        task = task_results.get(index, {})
        harness = task.get("harness") or _harness_for(
            scenario.hf_name, getattr(scenario, "draft_model", None)
        )
        run_dir = Path(task["run_dir"]) if task.get("run_dir") else None
        result_path = (
            _result_artifact_path(run_dir, harness) if run_dir is not None else None
        )
        data: dict = {}
        if result_path is not None and result_path.is_file():
            try:
                loaded = json.loads(result_path.read_text())
                data = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                pass
        model = data.get("model") or task.get("name") or scenario.hf_name
        status = results.get(index, task.get("status", "?"))
        draft_model = getattr(scenario, "draft_model", None)
        models.append(
            {
                "index": index,
                "model": model,
                **({"draft_model": draft_model} if draft_model else {}),
                "harness": harness,
                "reference": _reference_name(harness or ""),
                "tp": data.get("tp") or task.get("tp") or scenario.tp,
                "dtype": scenario.dtype,
                "workloads": list(_scenario_workloads(scenario)),
                "status": status,
                "paths": {
                    "run_log": str(run_dir / "run.log") if run_dir else None,
                    "results_json": str(result_path) if result_path else None,
                },
            }
        )
        if data and harness:
            throughput.extend(_throughput_rows_for_result(model, harness, data))
            latency.extend(_latency_rows_for_result(model, harness, data))
    passed = sum(model["status"].startswith("PASS") for model in models)
    return {
        "run": {
            "status": (
                "PASS"
                if models and passed == len(models)
                else "FAIL"
            ),
            "root": str(root),
            "models_total": len(models),
            "models_passed": passed,
        },
        "models": models,
        "throughput": throughput,
        "latency": latency,
    }


def _write_summary(root: Path, scenarios, results: dict[int, str]) -> None:
    summary = _build_summary(root, scenarios, results)
    path = root / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary saved to: {path}")
    if summary["throughput"]:
        print("\nTHROUGHPUT / E2E WORKLOADS")
        print(
            _format_table(
                summary["throughput"],
                [
                    ("model", "MODEL"),
                    ("workload", "WORKLOAD"),
                    ("reference", "REFERENCE"),
                    ("speedup", "SPEEDUP"),
                    ("correctness", "CORRECTNESS"),
                ],
            )
        )
    if summary["latency"]:
        print("\nLATENCY WORKLOADS")
        print(
            _format_table(
                summary["latency"],
                [
                    ("model", "MODEL"),
                    ("workload", "WORKLOAD"),
                    ("reference", "REFERENCE"),
                    ("speedup", "SPEEDUP"),
                ],
            )
        )


def _print_summary(scenarios, results: dict[int, str]) -> int:
    print(_c("\nvalidate summary", "1"))
    for index, scenario in enumerate(scenarios):
        status = results.get(index, "?")
        if status.startswith("PASS"):
            mark = _c("PASS", "32")
        elif status.startswith("SKIP"):
            mark = _c("SKIP", "2")
        else:
            mark = _c("FAIL", "31")
        print(f"  {mark:16} {scenario.hf_name}  [{status}]")
    return 0 if results and all(v.startswith("PASS") for v in results.values()) else 1


def _cancel_refs(ray, refs: list) -> None:
    for ref in refs:
        try:
            ray.cancel(ref, force=False)
        except Exception:
            pass
    if refs:
        try:
            _, pending = ray.wait(refs, num_returns=len(refs), timeout=10)
        except Exception:
            pending = refs
        for ref in pending:
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass


def _init_ray(ray, args):
    kwargs = {
        "ignore_reinit_error": True,
        "log_to_driver": False,
    }
    if args.ray_address:
        return ray.init(address=args.ray_address, **kwargs)
    try:
        return ray.init(
            include_dashboard=True,
            dashboard_host="127.0.0.1",
            **kwargs,
        )
    except Exception as exc:
        print(
            f"warning: Ray dashboard startup failed ({exc}); retrying without it",
            file=sys.stderr,
        )
        ray.shutdown()
        return ray.init(include_dashboard=False, **kwargs)


def run_validation(scenarios, args, gpus: list[str], root: Path) -> int:
    if not gpus:
        print("error: no GPUs selected for validation", file=sys.stderr)
        return 2

    timeout = int(
        os.environ.get("FASTKERNELS_VALIDATE_TIMEOUT_SEC", str(args.timeout))
    )
    jobs, results, cached = _plan_jobs(scenarios, args, len(gpus), root)
    root.mkdir(parents=True, exist_ok=True)
    scenario_name = _scenario_label(args.scenarios)
    for result in cached:
        task_name = _ray_job_id(scenario_name, result)
        _append_run_event(
            root,
            {
                "event": "task_finished",
                "task_name": task_name,
                "status": result["status"],
                "result": result,
            },
        )
    if not jobs:
        rc = _print_summary(scenarios, results)
        _append_run_event(
            root,
            {
                "event": "run_finished",
                "status": "PASS" if rc == 0 else "FAIL",
                "results": results,
            },
        )
        _write_summary(root, scenarios, results)
        return rc

    try:
        import ray
    except ModuleNotFoundError:
        print(
            "error: Ray is not installed. Install fastkernels dependencies or "
            "`pip install 'ray[default]'`.",
            file=sys.stderr,
        )
        return 2

    previous_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    refs: list = []
    initialized = False
    interrupted = False
    try:
        _init_ray(ray, args)
        initialized = True
        dashboard_url = _dashboard_url(ray)
        cluster_resources = ray.cluster_resources()
        print(
            f"\n{_c('>', '36')} submitting {len(jobs)} named Ray task(s) "
            f"across {len(gpus)} GPU(s)"
        )
        print(f"  Ray dashboard: {dashboard_url or 'unavailable'}")
        print(f"  output root: {root}")
        print(
            f"  cache root: {_run_cache_root(root)}"
            f"{'  (reused)' if args.resume else '  (fresh)'}"
        )
        print(f"  run log: {root / 'run.jsonl'}\n", flush=True)
        _append_run_event(
            root,
            {
                "event": "run_start",
                "scenario_table": str(args.scenarios),
                "visible_gpus": gpus,
                "dashboard_url": dashboard_url,
                "job_count": len(jobs),
                "cache_root": str(_run_cache_root(root)),
                "cache_reused": bool(args.resume),
            },
        )

        remote_fn = ray.remote(_ray_run_job)
        ref_to_job: dict[object, dict] = {}
        for job in jobs:
            task_name = _ray_job_id(scenario_name, job)
            resources = _ray_resource_options(
                job["tp"],
                len(gpus),
                cluster_resources,
            )
            ref = remote_fn.options(name=task_name, **resources).remote(
                job,
                timeout,
                args.stall_timeout,
                str(_REPO_ROOT),
                gpus,
                args.numactl_mode,
            )
            refs.append(ref)
            ref_to_job[ref] = job
            _append_run_event(
                root,
                {
                    "event": "task_submitted",
                    "task_name": task_name,
                    "job": job,
                    "resources": resources,
                },
            )
            print(
                f"{_c('>', '36')} {task_name}  tp={job['tp']}  "
                f"log={job['log_path']}",
                flush=True,
            )

        pending = list(refs)
        last_heartbeat = time.monotonic()
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=2.0)
            if not ready:
                now = time.monotonic()
                if now - last_heartbeat >= _RAY_PROGRESS_INTERVAL_SEC:
                    print(f"  running/pending tasks: {len(pending)}", flush=True)
                    last_heartbeat = now
                continue
            for ref in ready:
                job = ref_to_job[ref]
                task_name = _ray_job_id(scenario_name, job)
                try:
                    result = ray.get(ref)
                    status = result["status"]
                except Exception as exc:
                    status = f"FAIL(ray:{type(exc).__name__})"
                    result = {
                        **job,
                        "status": status,
                        "elapsed_s": 0.0,
                        "error": repr(exc),
                    }
                results[job["index"]] = status
                _append_run_event(
                    root,
                    {
                        "event": "task_finished",
                        "task_name": task_name,
                        "status": status,
                        "result": result,
                    },
                )
                mark = _c("PASS", "32") if status == "PASS" else _c("FAIL", "31")
                print(
                    f"{mark} {task_name}  {status} "
                    f"({int(result.get('elapsed_s', 0))}s)  "
                    f"{result.get('log_path', job['log_path'])}",
                    flush=True,
                )
                if status != "PASS":
                    for line in _tail(Path(job["log_path"]), 12).splitlines():
                        print(f"    {_c(line, '2')}")
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupt: cancelling Ray validation tasks", file=sys.stderr)
        _cancel_refs(ray, refs)
    finally:
        if args.gpus:
            if previous_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cvd
        if initialized:
            ray.shutdown()

    if interrupted:
        for index in range(len(scenarios)):
            results.setdefault(index, "FAIL(cancelled)")
    rc = _print_summary(scenarios, results)
    _append_run_event(
        root,
        {
            "event": "run_finished",
            "status": "PASS" if rc == 0 else "FAIL",
            "results": results,
        },
    )
    _write_summary(root, scenarios, results)
    return 130 if interrupted else rc
