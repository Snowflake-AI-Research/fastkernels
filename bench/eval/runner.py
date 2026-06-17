"""Eval runner: async execution of E2E job pairs with GPU pool management.

Runs baseline + candidate subprocess pairs for each (model, tp) job
in the eval plan. TP-aware scheduling: TP=1 jobs run N_GPU in parallel,
TP=4 jobs run N_GPU/4 in parallel, etc.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from kb_nano.bench.utils.worker import KB_NANO_MULTI_SCENARIO_WORKER, run_worker

from .aggregator import Aggregator, EvalReport
from .config import EvalConfig
from .planner import EvalJob, EvalPlan, EvalPlanner


@dataclass
class JobResult:
    """Result from running a single (model, tp) baseline+candidate pair."""
    model: str
    tp: int
    category: str
    status: str = "OK"
    error: str | None = None

    throughput_results: list[dict] = field(default_factory=list)
    latency_results: list[dict] = field(default_factory=list)


def _run_subprocess(
    job: EvalJob,
    no_candidate_kernels: bool,
    max_seq_len: int,
) -> dict | None:
    """Run a single E2E subprocess (baseline or candidate)."""
    from kb_nano import KB_ROOT, PROJECT_ROOT
    kb_root = str(PROJECT_ROOT)
    package_name = KB_ROOT.name

    label_suffix = "baseline" if no_candidate_kernels else "candidate"
    config = {
        "model": job.model,
        "tp": job.tp,
        "seed": job.seed,
        "temperature": job.temperature,
        "enforce_eager": job.enforce_eager,
        "max_model_len": max_seq_len,
        "project_root": kb_root,
        "package_name": package_name,
        "no_candidate_kernels": no_candidate_kernels,
        "scenarios": job.throughput_data,
        "latency_scenarios": job.latency_data,
    }

    return run_worker(
        KB_NANO_MULTI_SCENARIO_WORKER, config,
        f"{label_suffix} [{job.short_name}] (TP={job.tp})",
    )


def _run_job_pair(job: EvalJob, max_seq_len: int) -> JobResult:
    """Run baseline + candidate for a single (model, tp) pair."""
    result = JobResult(
        model=job.model, tp=job.tp, category=job.category,
    )

    baseline_raw = _run_subprocess(job, no_candidate_kernels=True, max_seq_len=max_seq_len)
    if baseline_raw is None:
        result.status = "FAILED"
        result.error = "Baseline subprocess failed"
        return result

    candidate_raw = _run_subprocess(job, no_candidate_kernels=False, max_seq_len=max_seq_len)
    if candidate_raw is None:
        result.status = "FAILED"
        result.error = "Candidate subprocess failed"
        return result

    baseline_throughput = baseline_raw.get("throughput", [])
    candidate_throughput = candidate_raw.get("throughput", [])

    for bl, cd in zip(baseline_throughput, candidate_throughput):
        bl_tps = bl["total_output_tokens"] / bl["elapsed"] if bl["elapsed"] > 0 else 0
        cd_tps = cd["total_output_tokens"] / cd["elapsed"] if cd["elapsed"] > 0 else 0
        speedup = cd_tps / bl_tps if bl_tps > 0 else float("inf")

        aligned_matches, aligned_total = 0, 0
        if "outputs" in bl and "outputs" in cd and job.temperature == 0.0:
            aligned_total = min(len(bl["outputs"]), len(cd["outputs"]))
            for b_out, c_out in zip(bl["outputs"], cd["outputs"]):
                if b_out.get("token_ids") == c_out.get("token_ids"):
                    aligned_matches += 1

        result.throughput_results.append({
            "name": bl.get("name", "unknown"),
            "baseline_tok_s": bl_tps,
            "candidate_tok_s": cd_tps,
            "speedup": speedup,
            "baseline_elapsed": bl["elapsed"],
            "candidate_elapsed": cd["elapsed"],
            "aligned_matches": aligned_matches,
            "aligned_total": aligned_total,
        })

    baseline_latency = baseline_raw.get("latency", [])
    candidate_latency = candidate_raw.get("latency", [])

    import numpy as np
    for bl_lat, cd_lat in zip(baseline_latency, candidate_latency):
        bl_med = float(np.median(bl_lat["latencies"]))
        cd_med = float(np.median(cd_lat["latencies"]))
        speedup = bl_med / cd_med if cd_med > 0 else float("inf")
        result.latency_results.append({
            "name": bl_lat.get("name", "unknown"),
            "baseline_median": bl_med,
            "candidate_median": cd_med,
            "speedup": speedup,
            "batch_size": bl_lat.get("batch_size", 0),
            "input_len": bl_lat.get("input_len", 0),
            "output_len": bl_lat.get("output_len", 0),
        })

    return result


def _group_jobs_by_tp(jobs: list[EvalJob]) -> dict[int, list[EvalJob]]:
    """Group jobs by TP degree for scheduling."""
    groups: dict[int, list[EvalJob]] = {}
    for job in jobs:
        groups.setdefault(job.tp, []).append(job)
    return groups


async def run_eval(config: EvalConfig, gpu_pool: int = 8) -> EvalReport:
    """Run the full eval sweep.

    Generates the eval plan, executes all job pairs, and aggregates results.
    Jobs are run sequentially per TP group (parallel execution requires
    GPU isolation which is handled by the subprocess model).
    """
    planner = EvalPlanner(config)
    plan = planner.plan()

    if not plan.jobs:
        print("No jobs to run.")
        return Aggregator.aggregate([])

    print("=" * 70)
    print("  Evaluation")
    print("=" * 70)
    categories = sorted(set(j.category for j in plan.jobs))
    print(f"  Categories     : {', '.join(categories)}")
    print(f"  Models         : {len(set(j.model for j in plan.jobs))}")
    print(f"  TP degrees     : {sorted(set(j.tp for j in plan.jobs))}")
    print(f"  Throughput     : WildChat prefill-heavy, balanced, decode-heavy")
    print(f"                   ({config.num_prompts} real requests each, model chat template)")
    print(f"  Latency        : single-request (bs=1, 128/128),")
    print(f"                   fixed-batch-32 (bs=32, 128/128)")
    print("=" * 70)

    start_time = time.perf_counter()
    results: list[JobResult] = []

    tp_groups = _group_jobs_by_tp(plan.jobs)
    for tp_degree in sorted(tp_groups):
        group_jobs = tp_groups[tp_degree]
        parallel_slots = max(1, gpu_pool // tp_degree)

        print(f"\n  TP={tp_degree}: {len(group_jobs)} job(s), "
              f"up to {parallel_slots} parallel slot(s)")

        for job in group_jobs:
            print(f"\n  Running {job.short_name} (TP={job.tp})...")
            job_result = _run_job_pair(job, plan.max_seq_len)
            results.append(job_result)

            if job_result.status == "FAILED":
                print(f"  FAILED: {job_result.error}")
            else:
                for tr in job_result.throughput_results:
                    print(f"    {tr['name']}: {tr['speedup']:.2f}x speedup")

    wall_clock = time.perf_counter() - start_time
    report = Aggregator.aggregate(results, wall_clock_seconds=wall_clock)
    return report
