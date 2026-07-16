"""Evaluate candidate task implementations against the baseline, end-to-end.

Runs the models/workloads named by a scenarios table (the same input
``fastkernels capture`` takes -- a path to a YAML, or a packaged name like
``full`` / ``default`` / ``minimal`` resolved against ``fastkernels/scenarios/``)
through the fastkernels ``LlamaEngine`` **twice per scenario**: once with the
stock ``tasks/baseline`` operators and once with the user's ``tasks/candidate``
operators swapped in. For every workload it reports

  * **correctness** -- the candidate's greedy token ids must match the
    baseline's exactly (``temperature=0``); any request that diverges fails the
    scenario, and
  * **performance** -- candidate-vs-baseline throughput (tok/s) and latency
    (median seconds) with the resulting speedup.

``--self-test`` uses the baseline implementation *as* the candidate: the two
runs are identical, so correctness must be a perfect 100% match and the speedup
~1.0x. It exercises the harness itself (data loading, scheduling, alignment,
timing) without needing any candidate kernels.

``--max-requests`` caps the number of prompts loaded per workload and
``--max-layers`` builds and runs only the first N transformer decoder layers of
each model (embeddings / final norm / LM head untouched) -- both exactly as in
``fastkernels capture``, to bound the cost of a smoke run.

Scenarios are evaluated in parallel across the available GPUs: each scenario
runs in its own child process pinned to a private set of GPUs (a ``tp=N``
scenario claims N GPUs) via ``CUDA_VISIBLE_DEVICES``, so a crash / OOM / CUDA
fault in one never brings down the others. Use ``--gpus`` to restrict the pool.

Usage::

    python -m fastkernels eval full                       # baseline vs candidate
    python -m fastkernels eval minimal --self-test        # harness self-test
    python -m fastkernels eval default --max-requests 8 --max-layers 4 --gpus 0

Candidate discovery and class swapping are done here (no dependency on the
retired ``infra.kernel_swapper``): the operator<->class map comes from
``fastkernels.list.discover_operator_targets`` (pure static analysis), each
operator's candidate is loaded from ``tasks/candidate/L<level>/<name>.py`` when
present, and the baseline class references are monkey-patched in-process before
the candidate engine is built. For ``tp>1`` the same swap is re-applied inside
every spawned tensor-parallel worker via an env-gated ``sitecustomize``.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from .capture import (
    _detect_gpu_ids,
    _engine_dtype,
    _is_eagle3,
    _is_fla,
    _is_jamba,
    _kill_process_group,
    _print_log_tail,
    _scenario_log_name,
    _wait_any,
    _NCCL_PORT_BASE,
)
from .list import apply_candidates, discover_candidate_impls, restore_candidates
from .workloads import Purpose, load_real_prompt_workload, resolve_benchmark, spec_for

# Default directory for eval reports (override per-run with ``--output``).
EVAL_DIR = Path.home() / ".fastkernels" / "evals"

# Internal env var the parallel scheduler uses to tell a worker subprocess which
# scenario to evaluate (set alongside CUDA_VISIBLE_DEVICES by the parent). Not a
# user-facing flag.
_WORKER_INDEX_ENV = "FK_EVAL_WORKER_INDEX"

# Env var that tells a spawned process (a tensor-parallel worker) to swap the
# candidate operators in at startup, via the installed sitecustomize. The eval
# child sets it only around the *candidate* engine build.
_APPLY_CANDIDATES_ENV = "FASTKERNELS_EVAL_APPLY_CANDIDATES"

# ---------------------------------------------------------------------------
# Candidate discovery + class swapping
#
# The discovery + monkey-patch helpers live in ``fastkernels.list`` (built on
# its static operator<->class map) so the engine / server / eval all share one
# implementation. eval imports ``discover_candidate_impls`` / ``apply_candidates``
# / ``restore_candidates`` above; ``tp>1`` propagation is handled by the
# sitecustomize below, which calls ``fastkernels.list._apply_candidates_from_env``
# inside every spawned tensor-parallel worker.
# ---------------------------------------------------------------------------
def _install_candidate_sitecustomize() -> None:
    """Install a sitecustomize that re-applies the candidate swap in every
    spawned process (the engine's ``tp>1`` tensor-parallel workers start fresh
    interpreters that would otherwise run the baseline). Gated by
    ``_APPLY_CANDIDATES_ENV`` so it is inert until the candidate phase sets it.
    """
    site_dir = Path(os.environ.get(
        "FASTKERNELS_EVAL_SITECUSTOMIZE_DIR",
        "/tmp/fastkernels_eval_sitecustomize",
    ))
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(
        "import os\n"
        f"if os.environ.get({_APPLY_CANDIDATES_ENV!r}):\n"
        "    try:\n"
        "        import fastkernels.list as _l\n"
        "        _l._apply_candidates_from_env()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    current = os.environ.get("PYTHONPATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if str(site_dir) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(site_dir), *parts])


# ---------------------------------------------------------------------------
# Workload data loading (text LLM path)
# ---------------------------------------------------------------------------
def _model_max_ctx(model_name: str) -> int | None:
    """The model's ``max_position_embeddings`` (top-level or nested text config),
    or ``None`` when it cannot be read."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        val = getattr(cfg, "max_position_embeddings", None)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return None


def _load_scenario_runs(scenario, args, tokenizer):
    """Build the per-workload prompt data for one scenario (text workloads only).

    Returns ``(throughput_runs, latency_runs, global_max_seq_len, skipped)``.
    ``skipped`` lists ``(workload_name, reason)`` for non-text workloads we did
    not evaluate.
    """
    model_ctx = _model_max_ctx(scenario.hf_name)

    def _fit(prompt_ids: list[list[int]], output_lens: list[int]):
        """Drop requests whose prompt+decode exceeds the model context."""
        if model_ctx is None:
            return prompt_ids, output_lens
        kept_p, kept_o = [], []
        for p, ol in zip(prompt_ids, output_lens):
            if len(p) + ol <= model_ctx:
                kept_p.append(p)
                kept_o.append(ol)
        return kept_p, kept_o

    throughput_runs: list[dict] = []
    latency_runs: list[dict] = []
    skipped: list[tuple[str, str]] = []
    global_max_seq_len = 0

    for i, wl in enumerate(scenario.throughput_workloads):
        spec = spec_for(wl)
        params = spec.params
        modality = getattr(params, "modality", "text")
        if modality != "text":
            skipped.append((wl.value, f"modality={modality} (correctness "
                                      "comparison is text-only)"))
            continue
        n_req = min(args.max_requests, getattr(params, "num_requests", args.max_requests))
        samples = load_real_prompt_workload(
            wl.value, tokenizer, num_requests=n_req,
            dataset_name=getattr(params, "dataset_name", "") or None,
            decode_cap=getattr(params, "decode_cap", None),
            seed=args.seed + i,
        )
        prompt_ids = [list(s.prompt_token_ids) for s in samples]
        output_lens = [s.output_len for s in samples]
        prompt_ids, output_lens = _fit(prompt_ids, output_lens)
        if not prompt_ids:
            skipped.append((wl.value, "all prompts exceed the model context"))
            continue
        global_max_seq_len = max(
            global_max_seq_len,
            max(len(p) + ol for p, ol in zip(prompt_ids, output_lens)),
        )
        throughput_runs.append({
            "name": wl.value,
            "prompt_token_ids": prompt_ids,
            "output_lens": output_lens,
        })

    for j, wl in enumerate(scenario.latency_workloads):
        spec = spec_for(wl)
        params = spec.params
        modality = getattr(params, "modality", "text")
        if modality != "text":
            skipped.append((wl.value, f"modality={modality} (correctness "
                                      "comparison is text-only)"))
            continue
        bs = min(args.max_requests, getattr(params, "batch_size", 1))
        output_len = getattr(params, "output_len", 128)
        samples = load_real_prompt_workload(
            "mixed", tokenizer, num_requests=bs, decode_cap=output_len,
            dataset_name=getattr(params, "dataset_name", "") or None,
            seed=args.seed + 100 + j,
        )
        prompt_ids = [list(s.prompt_token_ids) for s in samples]
        prompt_ids, _ = _fit(prompt_ids, [output_len] * len(prompt_ids))
        if not prompt_ids:
            skipped.append((wl.value, "all prompts exceed the model context"))
            continue
        global_max_seq_len = max(
            global_max_seq_len,
            max(len(p) + output_len for p in prompt_ids),
        )
        latency_runs.append({
            "name": wl.value,
            "prompt_token_ids": prompt_ids,
            "input_len": max(len(p) for p in prompt_ids),
            "output_len": output_len,
            "batch_size": len(prompt_ids),
            "num_warmup": getattr(params, "num_warmup", 3),
            "num_iters": getattr(params, "num_iters", 5),
        })

    if model_ctx is not None and global_max_seq_len > model_ctx:
        global_max_seq_len = model_ctx
    return throughput_runs, latency_runs, global_max_seq_len, skipped


# ---------------------------------------------------------------------------
# Running one implementation (baseline or candidate) of a scenario
# ---------------------------------------------------------------------------
def _run_impl(scenario, args, throughput_runs, latency_runs, max_seq_len):
    """Build a fresh engine (baseline or candidate, per the currently-patched
    classes) and run every workload. Returns per-workload throughput + latency
    results with the generated token ids for correctness comparison."""
    mod = __import__("fastkernels.infra.engine",
                     fromlist=["LlamaEngine", "SamplingParams"])
    LlamaEngine, SamplingParams = mod.LlamaEngine, mod.SamplingParams

    engine_kwargs = dict(
        model_name=scenario.hf_name,
        dtype=_engine_dtype(scenario.dtype),
        seed=args.seed,
        tensor_parallel_size=scenario.tp,
        enforce_eager=args.enforce_eager or scenario.enforce_eager,
        max_num_seqs=scenario.max_num_seqs,
        max_layers=args.max_layers,
    )
    if max_seq_len:
        engine_kwargs["max_model_len"] = max_seq_len
    engine = LlamaEngine(**engine_kwargs)

    throughput: list[dict] = []
    latency: list[dict] = []
    try:
        # Warmup so the first timed workload does not pay one-off costs.
        engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

        for run in throughput_runs:
            sp = [SamplingParams(temperature=args.temperature, top_p=1.0,
                                 max_tokens=ol, ignore_eos=True)
                  for ol in run["output_lens"]]
            engine.block_manager.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(run["prompt_token_ids"], sp,
                                      use_tqdm=False, decode_text=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            throughput.append({
                "name": run["name"],
                "elapsed": elapsed,
                "num_requests": len(outputs),
                "total_output_tokens": sum(len(o.token_ids) for o in outputs),
                "token_ids": [list(o.token_ids) for o in outputs],
            })

        for run in latency_runs:
            sp = SamplingParams(temperature=0.0, ignore_eos=True,
                                max_tokens=run["output_len"])
            prompts = run["prompt_token_ids"]
            for _ in range(run["num_warmup"]):
                engine.block_manager.reset()
                torch.cuda.synchronize()
                engine.generate(prompts, sp, use_tqdm=False)
                torch.cuda.synchronize()
            latencies = []
            for _ in range(run["num_iters"]):
                engine.block_manager.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                engine.generate(prompts, sp, use_tqdm=False)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)
            latency.append({
                "name": run["name"],
                "batch_size": run["batch_size"],
                "input_len": run["input_len"],
                "output_len": run["output_len"],
                "latencies": latencies,
            })
    finally:
        # The engine registers ``_cleanup`` with atexit (a strong ref), so a
        # plain del would not free its GPU memory before the next engine builds.
        engine._cleanup()
        atexit.unregister(engine._cleanup)
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"throughput": throughput, "latency": latency}


def _median(values: list[float]) -> float:
    import numpy as np
    return float(np.median(values)) if values else 0.0


def _alignment(baseline_ids: list[list[int]], candidate_ids: list[list[int]]) -> dict:
    """Per-request greedy-token-id comparison of two runs."""
    total = min(len(baseline_ids), len(candidate_ids))
    exact = matched_tokens = total_tokens = 0
    for a, b in zip(baseline_ids, candidate_ids):
        n = max(len(a), len(b))
        total_tokens += n
        if a == b:
            exact += 1
            matched_tokens += len(a)
        else:
            matched_tokens += sum(1 for x, y in zip(a, b) if x == y)
    return {
        "exact_matches": exact,
        "total_requests": total,
        "matched_tokens": matched_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Report path + output
# ---------------------------------------------------------------------------
def _scenario_slug(scenario, args) -> str:
    model = scenario.hf_name.replace("/", "__")
    tag = "_selftest" if args.self_test else ""
    layers = f"_L{args.max_layers}" if args.max_layers is not None else ""
    return f"{model}_tp{scenario.tp}_{scenario.dtype}{layers}{tag}"


def _report_path(scenario, args, multi: bool) -> Path:
    slug = _scenario_slug(scenario, args)
    if args.output is not None:
        p = Path(args.output)
        return p.with_name(f"{p.stem}_{slug}{p.suffix or '.json'}") if multi else p
    return EVAL_DIR / f"{slug}.json"


# ---------------------------------------------------------------------------
# Evaluate one scenario (baseline vs candidate), in-process
# ---------------------------------------------------------------------------
def _eval_scenario(scenario, args, multi: bool) -> tuple[Path | None, bool]:
    """Run baseline + candidate for a scenario, compare, write a report.

    Returns ``(report_path_or_None, ok)`` where ``ok`` is False if the scenario
    failed to run or any workload's correctness check did not pass.
    """
    if _is_eagle3(scenario) or _is_fla(scenario) or _is_jamba(scenario):
        print(f"  !! SKIP {scenario.hf_name}: EAGLE-3/FLA/Jamba use specialized "
              f"engines; eval covers the standard LlamaEngine path.")
        return None, True

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(scenario.hf_name, trust_remote_code=True)

    throughput_runs, latency_runs, max_seq_len, skipped = _load_scenario_runs(
        scenario, args, tokenizer)
    for name, reason in skipped:
        print(f"  (skipped workload {name}: {reason})")
    if not throughput_runs and not latency_runs:
        print(f"  !! SKIP {scenario.hf_name}: no text workloads to evaluate.")
        return None, True

    # Is there anything to swap? Self-test deliberately keeps candidate ==
    # baseline; otherwise discover the candidate operators (informational here,
    # applied below / in spawned workers).
    candidate_pairs = [] if args.self_test else discover_candidate_impls()
    compare_candidate = bool(candidate_pairs)

    print(f"\n########## Scenario: {scenario.hf_name} "
          f"(tp={scenario.tp}, dtype={scenario.dtype}"
          f"{f', max_layers={args.max_layers}' if args.max_layers else ''}) "
          f"##########")
    if args.self_test:
        print("  Mode: --self-test (candidate = baseline; expect 100% match, ~1.0x)")
    elif compare_candidate:
        print(f"  Candidate operators: "
              f"{', '.join(f'L{t.level}:{t.name}' for t, _, _ in candidate_pairs)}")
    else:
        print("  NOTE: no candidate implementations found under tasks/candidate/; "
              "running baseline twice (nothing to compare).")

    print(f"  Workloads: throughput={[r['name'] for r in throughput_runs]} "
          f"latency={[r['name'] for r in latency_runs]}")

    # --- baseline run (stock operators; env must be unset so spawned TP workers
    #     stay on the baseline too) ---
    os.environ.pop(_APPLY_CANDIDATES_ENV, None)
    print("  Running baseline ...")
    baseline = _run_impl(scenario, args, throughput_runs, latency_runs, max_seq_len)

    # --- candidate run ---
    undo: list[tuple] = []
    if compare_candidate:
        os.environ[_APPLY_CANDIDATES_ENV] = "1"   # spawned TP workers self-apply
        undo = apply_candidates(candidate_pairs)  # this (rank-0) process
        print("  Running candidate ...")
    else:
        print("  Running candidate (== baseline) ...")
    try:
        candidate = _run_impl(scenario, args, throughput_runs, latency_runs, max_seq_len)
    finally:
        if undo:
            restore_candidates(undo)
        os.environ.pop(_APPLY_CANDIDATES_ENV, None)

    # --- compare ---
    all_correct = True
    tp_results = []
    for bl, cd in zip(baseline["throughput"], candidate["throughput"]):
        bl_tps = bl["total_output_tokens"] / bl["elapsed"] if bl["elapsed"] else 0.0
        cd_tps = cd["total_output_tokens"] / cd["elapsed"] if cd["elapsed"] else 0.0
        align = _alignment(bl["token_ids"], cd["token_ids"])
        correct = align["exact_matches"] == align["total_requests"]
        all_correct = all_correct and correct
        tp_results.append({
            "name": bl["name"],
            "num_requests": bl["num_requests"],
            "baseline_tok_per_s": bl_tps,
            "candidate_tok_per_s": cd_tps,
            "speedup": (cd_tps / bl_tps) if bl_tps else 0.0,
            "alignment": align,
            "correct": correct,
        })

    lat_results = []
    for bl, cd in zip(baseline["latency"], candidate["latency"]):
        bl_med = _median(bl["latencies"])
        cd_med = _median(cd["latencies"])
        lat_results.append({
            "name": bl["name"],
            "batch_size": bl["batch_size"],
            "input_len": bl["input_len"],
            "output_len": bl["output_len"],
            "baseline_median_s": bl_med,
            "candidate_median_s": cd_med,
            "speedup": (bl_med / cd_med) if cd_med else 0.0,
        })

    report = {
        "model": scenario.hf_name,
        "tp": scenario.tp,
        "dtype": scenario.dtype,
        "self_test": args.self_test,
        "compared_candidate": compare_candidate,
        "max_layers": args.max_layers,
        "max_requests": args.max_requests,
        "temperature": args.temperature,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "correctness_passed": all_correct,
        "throughput": tp_results,
        "latency": lat_results,
    }
    out_path = _report_path(scenario, args, multi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_scenario_summary(scenario, tp_results, lat_results, all_correct)
    print(f"  Report written to {out_path}")
    return out_path, all_correct


def _print_scenario_summary(scenario, tp_results, lat_results, all_correct) -> None:
    print(f"\n  {'-' * 84}")
    print(f"  RESULTS: {scenario.hf_name} (tp={scenario.tp})   "
          f"correctness: {'PASS' if all_correct else 'FAIL'}")
    if tp_results:
        print(f"  {'THROUGHPUT':<16} {'REQS':>5} {'BASE tok/s':>12} "
              f"{'CAND tok/s':>12} {'SPEEDUP':>8} {'MATCH':>12} {'':>6}")
        for r in tp_results:
            a = r["alignment"]
            match = f"{a['exact_matches']}/{a['total_requests']}"
            print(f"  {r['name']:<16} {r['num_requests']:>5} "
                  f"{r['baseline_tok_per_s']:>12,.0f} {r['candidate_tok_per_s']:>12,.0f} "
                  f"{r['speedup']:>7.2f}x {match:>12} "
                  f"{'OK' if r['correct'] else 'DIFF':>6}")
    if lat_results:
        print(f"  {'LATENCY':<16} {'BS':>5} {'BASE med':>12} "
              f"{'CAND med':>12} {'SPEEDUP':>8}")
        for r in lat_results:
            print(f"  {r['name']:<16} {r['batch_size']:>5} "
                  f"{r['baseline_median_s']:>11.4f}s {r['candidate_median_s']:>11.4f}s "
                  f"{r['speedup']:>7.2f}x")
    print(f"  {'-' * 84}")


# ---------------------------------------------------------------------------
# Parallel GPU scheduler (one child process per scenario)
# ---------------------------------------------------------------------------
def _worker_command(args) -> list[str]:
    """Argv for a worker child. The scenario it evaluates is passed out-of-band
    via ``_WORKER_INDEX_ENV`` (set by ``_launch``); the ``scenarios`` positional
    is forwarded so the child resolves the same table + index ordering."""
    cmd = [
        sys.executable, "-u", "-m", "fastkernels.eval",
        args.scenarios,
        "--max-requests", str(args.max_requests),
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
    ]
    if args.self_test:
        cmd.append("--self-test")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.max_layers is not None:
        cmd += ["--max-layers", str(args.max_layers)]
    if args.output:
        cmd += ["--output", args.output]
    return cmd


def _run_scenarios_parallel(scenarios, args, gpu_ids: list[str], multi: bool) -> int:
    """Evaluate ``scenarios`` concurrently, packing them onto ``gpu_ids`` by TP."""
    import subprocess

    total = len(gpu_ids)
    log_dir = EVAL_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, tuple[str, str]] = {}
    pending: list[tuple[int, object]] = []
    for i, s in enumerate(scenarios):
        if s.tp > total:
            detail = f"needs tp={s.tp} > {total} GPU(s) available"
            results[i] = ("skipped", detail)
            print(f"  !! SKIP scenario[{i}] {s.hf_name}: {detail}")
        else:
            pending.append((i, s))
    pending.sort(key=lambda t: (-t[1].tp, t[0]))

    free = list(gpu_ids)
    running: dict = {}

    def _launch(i, s) -> None:
        assign = [free.pop(0) for _ in range(s.tp)]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(assign)
        env[_WORKER_INDEX_ENV] = str(i)
        env["FASTKERNELS_NCCL_PORT"] = str(_NCCL_PORT_BASE + i)
        # Never leak the candidate-swap trigger into a child's own startup; it is
        # set by the child itself only around its candidate engine build.
        env.pop(_APPLY_CANDIDATES_ENV, None)
        log_path = log_dir / f"{_scenario_log_name(s, i)}.log"
        logf = open(log_path, "w")
        proc = subprocess.Popen(
            _worker_command(args), stdout=logf,
            stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
        running[proc] = {
            "index": i, "scenario": s, "gpus": assign,
            "log": log_path, "logf": logf,
            "pgid": proc.pid, "start": time.monotonic(),
        }
        print(f"  -> [GPU {env['CUDA_VISIBLE_DEVICES']}] scenario[{i}] "
              f"{s.hf_name} (tp={s.tp}) started; log {log_path}")

    try:
        while pending or running:
            made_progress = True
            while made_progress:
                made_progress = False
                for pos, (i, s) in enumerate(pending):
                    if s.tp <= len(free):
                        _launch(i, s)
                        pending.pop(pos)
                        made_progress = True
                        break
            if not running:
                for i, s in pending:
                    results[i] = ("skipped", "could not be scheduled")
                break
            for proc in _wait_any(running):
                info = running.pop(proc)
                info["logf"].close()
                free.extend(info["gpus"])
                i, s, rc = info["index"], info["scenario"], proc.returncode
                killed = info.get("killed_reason")
                if killed:
                    results[i] = ("error", f"{killed}; {info['log']}")
                elif rc == 0:
                    results[i] = ("ok", "")
                elif rc == 1:
                    results[i] = ("failed", str(info["log"]))
                else:
                    results[i] = ("error", f"exit={rc}; {info['log']}")
                print(f"  <- [{results[i][0].upper()}] scenario[{i}] {s.hf_name} "
                      f"(rc={rc}); freed GPU {','.join(info['gpus'])}")
                if results[i][0] == "error":
                    _print_log_tail(info["log"])
    finally:
        for proc, info in list(running.items()):
            _kill_process_group(proc, info["pgid"])
            info["logf"].close()

    print("\nEval summary:")
    ok_all = True
    for i, s in enumerate(scenarios):
        state, detail = results.get(i, ("unknown", ""))
        if state not in ("ok", "skipped"):
            ok_all = False
        line = f"  [{state.upper()}] scenario[{i}] {s.hf_name} (tp={s.tp})"
        if detail:
            line += f" -- {detail}"
        print(line)
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastkernels eval",
        description=(
            "Evaluate candidate task implementations against the baseline "
            "end-to-end: run the scenarios' models/workloads through the "
            "fastkernels engine with baseline vs candidate operators and report "
            "correctness (greedy token-id match) and performance (throughput / "
            "latency speedup)."
        ),
    )
    parser.add_argument(
        "scenarios",
        help="Path to a scenarios YAML, or a packaged name resolved against "
             "fastkernels/scenarios/ (e.g. 'full', 'default', 'minimal').",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Use the baseline implementation as the candidate. The two runs "
             "are identical, so correctness must be a perfect match and the "
             "speedup ~1.0x -- a check of the harness itself.",
    )
    parser.add_argument(
        "--max-requests", type=int, default=1_000_000,
        help="Max prompts to load per workload (default: every available row).",
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Build and run only the first MAX_LAYERS transformer decoder "
             "layers of each model (embeddings / final norm / LM head "
             "untouched), to bound the cost of a smoke run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default 0.0: greedy, for deterministic "
             "correctness comparison).",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable CUDA graphs (per-scenario setting otherwise).")
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated physical GPU ids to schedule across (default: all "
             "visible GPUs). Scenarios are packed onto these by TP degree and "
             "evaluated in parallel, each in its own GPU-pinned subprocess.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Base path for the JSON report(s) (default: "
             "~/.fastkernels/evals/<scenario-slug>.json per scenario).",
    )
    args = parser.parse_args(argv)
    if args.max_layers is not None and args.max_layers < 1:
        parser.error("--max-layers must be >= 1")

    try:
        scenarios = resolve_benchmark(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  !! could not load scenarios {args.scenarios!r}: {exc}")
        return 2
    if not scenarios:
        print(f"  !! no scenarios in {args.scenarios!r}")
        return 2

    multi = len(scenarios) > 1

    # --- Worker mode: evaluate exactly the one scenario the parent assigned. ---
    worker_index = os.environ.get(_WORKER_INDEX_ENV)
    if worker_index is not None:
        try:
            idx = int(worker_index)
        except ValueError:
            print(f"  !! invalid {_WORKER_INDEX_ENV}={worker_index!r}")
            return 2
        if not 0 <= idx < len(scenarios):
            print(f"  !! {_WORKER_INDEX_ENV}={idx} out of range "
                  f"(have {len(scenarios)} scenario(s))")
            return 2
        if any(s.tp > 1 for s in scenarios):
            _install_candidate_sitecustomize()
        try:
            _, ok = _eval_scenario(scenarios[idx], args, multi)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"\n  !! Scenario {scenarios[idx].hf_name} failed: {exc!r}")
            return 2
        return 0 if ok else 1

    # --- Parent: schedule across GPUs (default), or run in-process as a
    #     fallback when only one scenario / no GPU is available. ---
    print("=" * 70)
    print("  fastkernels eval -- baseline vs "
          f"{'baseline [self-test]' if args.self_test else 'candidate'}")
    print("=" * 70)
    print(f"  Scenarios   : {len(scenarios)} from {args.scenarios!r}")
    print(f"  Max requests: {args.max_requests}")
    if args.max_layers is not None:
        print(f"  Max layers  : {args.max_layers}")
    print(f"  Temperature : {args.temperature}")
    print("=" * 70)

    gpu_ids = _detect_gpu_ids(args.gpus)
    if len(scenarios) > 1 and len(gpu_ids) >= 1:
        # tp>1 candidate swaps must reach spawned workers; install once up front.
        if not args.self_test and any(s.tp > 1 for s in scenarios):
            _install_candidate_sitecustomize()
        print(f"Scheduling {len(scenarios)} scenario(s) across {len(gpu_ids)} "
              f"GPU(s) [{', '.join(gpu_ids)}] by TP degree ...")
        return _run_scenarios_parallel(scenarios, args, gpu_ids, multi)

    if not args.self_test and any(s.tp > 1 for s in scenarios):
        _install_candidate_sitecustomize()
    exit_code = 0
    for scenario in scenarios:
        try:
            _, ok = _eval_scenario(scenario, args, multi)
            if not ok:
                exit_code = 1
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"\n  !! Scenario {scenario.hf_name} failed: {exc!r}; "
                  f"continuing.")
            exit_code = 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
