"""Smoke-test one adapter on one scenario, sequentially on the visible GPU(s).

    python -m fastkernels.e2e.smoke --scenarios default --index 7 --out /tmp/smoke \
        [--max-requests 8] [--correctness-samples 4] [--workloads a,b] \
        [--candidates-dir DIR [--only L1:x,L2:y] [--exclude ...]] [--eager]

Runs, each through ``fastkernels.e2e.runner`` in a fresh process:
baseline -> noise (baseline again, scored against the baseline) -> optional candidate
(scored against the baseline). Prints timings, speedups and ``Adapter.compare`` summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_one(out: Path, name: str, scenarios: str, index: int, run: dict, env: dict,
            candidates: bool, timeout: int) -> dict:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    spec = {"scenarios": scenarios, "index": index, "candidates": candidates,
            "run": {**run, "out_dir": str(d)}, "result": str(d / "result.json")}
    (d / "spec.json").write_text(json.dumps(spec, indent=1))
    with open(d / "log.txt", "w") as log:
        try:
            subprocess.run([sys.executable, "-m", "fastkernels.e2e.runner", str(d / "spec.json")],
                           env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    res = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else \
        {"status": "no-result", "error": (d / "log.txt").read_text()[-3000:]}
    res["_dir"] = str(d)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="default")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-requests", type=int, default=8)
    ap.add_argument("--correctness-samples", type=int, default=4)
    ap.add_argument("--workloads", default=None)
    ap.add_argument("--candidates-dir", default=None)
    ap.add_argument("--only", default=None, help="L<n>:<stem>,... (default: the model's kernels)")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from fastkernels.e2e.adapters import adapter_for
    from fastkernels.e2e.candidates import kernels_for, prepare_set
    from fastkernels.workloads import resolve_benchmark

    scenario = resolve_benchmark(args.scenarios)[args.index]
    adapter = adapter_for(scenario)
    print(f"scenario {args.index}: {scenario.hf_name} (tp={scenario.tp}) -> adapter {adapter.name}")
    run = {"seed": 42, "max_requests": args.max_requests, "correctness_samples": args.correctness_samples,
           "enforce_eager": args.eager,
           "workloads": args.workloads.split(",") if args.workloads else None}
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("FASTKERNELS_CANDIDATE")}

    results = {}
    results["baseline"] = run_one(args.out, "baseline", args.scenarios, args.index,
                                  {**run, "reference": None}, base_env, False, args.timeout)
    ref = str(Path(results["baseline"]["_dir"]) / "outputs.pt")
    if not args.skip_noise:
        results["noise"] = run_one(args.out, "noise", args.scenarios, args.index,
                                   {**run, "reference": ref},
                                   {**base_env, "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"},
                                   False, args.timeout)
    if args.candidates_dir:
        set_dir = prepare_set(Path(args.candidates_dir), args.out / "sets")
        only = args.only or ",".join(kernels_for(set_dir, scenario.hf_name))
        env = {**base_env, "FASTKERNELS_CANDIDATE_DIR": str(set_dir),
               "FASTKERNELS_CANDIDATE_ONLY": only, "FASTKERNELS_CANDIDATE_EXCLUDE": args.exclude,
               "TORCH_EXTENSIONS_DIR": str(args.out / "torch_extensions" / set_dir.name)}
        results["candidate"] = run_one(args.out, "candidate", args.scenarios, args.index,
                                       {**run, "reference": ref}, env, True, args.timeout)

    base_t = results["baseline"].get("timings") or {}
    ref_out = torch.load(ref, weights_only=False) if Path(ref).is_file() else None
    report = {}
    for name, res in results.items():
        line = {"status": res["status"], "t_total_s": res.get("t_total_s"),
                "t_import_s": res.get("t_import_s"), "swapped": res.get("swapped"),
                "not_swapped": res.get("not_swapped")}
        if res["status"] != "ok":
            line["error"] = (res.get("error") or "")[-600:]
        else:
            sp = {}
            for wl, t in (res.get("timings") or {}).items():
                b = base_t.get(wl)
                if b and b["value"] and t["value"]:
                    sp[wl] = round(t["value"] / b["value"] if t["kind"] == "throughput"
                                   else b["value"] / t["value"], 4)
            line["speedup_vs_baseline"] = sp
            cand_out = Path(res["_dir"]) / "outputs.pt"
            if name != "baseline" and ref_out is not None and cand_out.is_file():
                cmp = adapter.compare(ref_out, torch.load(cand_out, weights_only=False))
                line["correctness"] = cmp["summary"]
                line["per_sample_d"] = [round(x, 4) for x in cmp["per_sample"]][:32]
        report[name] = line
    print("SMOKE_REPORT " + json.dumps(report))
    (args.out / "smoke_report.json").write_text(json.dumps(report, indent=1))
    return 0 if all(r["status"] == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
