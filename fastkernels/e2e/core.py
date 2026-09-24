"""``fastkernels e2e``: end-to-end evaluation of candidate kernel sets with drop-and-retry.

    fastkernels e2e default --sets DIR[,DIR...] --out OUT [--gpus 0,1,...]
                    [--scenario-indices 0,4] [--max-requests N] [--correctness-samples 64]

Per scenario (model): a baseline run, a noise run (baseline in a fresh process, scored
against the baseline) and, per candidate set, a chain of attempts:

1. swap in the set's kernels that this model uses (from the set manifest);
2. if the run crashes, drop the candidate kernel named innermost in the traceback and
   retry; if no candidate file is named, bisect the active kernels for a crashing one;
3. if the run succeeds but correctness is grossly broken (mean discrepancy above
   ``--broken-threshold``), bisect for the kernel responsible and drop it;
4. stop at the first attempt that runs and is not broken (the *deployable subset*), or
   when no kernels are left, or after ``--max-attempts``.

Every run is a separate ``fastkernels.e2e.runner`` process pinned to its own GPUs. Results
(one JSON per scenario x set, plus baseline/noise) go to OUT/results; re-running the same
command resumes (finished items are skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .candidates import culprits, kernels_for, prepare_set

_NCCL_BASE = int(os.environ.get("FASTKERNELS_NCCL_PORT_BASE", "29500"))


# ---------------------------------------------------------------------------
# GPU pool
# ---------------------------------------------------------------------------
class GPUPool:
    def __init__(self, gpus: list[str]):
        self.free = list(gpus)
        self.total = len(gpus)
        self.cv = threading.Condition()
        self._port = 0

    def lease(self, n: int) -> tuple[list[str], int]:
        if n > self.total:
            raise RuntimeError(f"needs {n} GPUs, pool has {self.total}")
        with self.cv:
            self.cv.wait_for(lambda: len(self.free) >= n)
            got, self.free = self.free[:n], self.free[n:]
            self._port += 1
            return got, _NCCL_BASE + (self._port % 500)

    def release(self, gpus: list[str]) -> None:
        with self.cv:
            self.free += gpus
            self.cv.notify_all()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _categorize(error: str, log: str) -> str:
    text = f"{error}\n{log[-20000:]}"
    if re.search(r"torch\._dynamo|Dynamo|ConstraintViolation|torch\._inductor|InternalTorchDynamoError",
                 text):
        return "compile"
    if "TIMEOUT" in error:
        return "timeout"
    return "runtime"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class E2E:
    def __init__(self, args):
        from fastkernels.workloads import resolve_benchmark
        self.args = args
        self.out = Path(args.out).resolve()
        self.scenarios = resolve_benchmark(args.scenarios)
        gpus = args.gpus.split(",") if args.gpus else _detect_gpus()
        self.pool = GPUPool(gpus)
        self.log_lock = threading.Lock()
        self.sets = []
        for d in args.sets.split(","):
            if d:
                self.sets.append(prepare_set(Path(d).resolve(), self.out / "work" / "sets"))

    # -- bookkeeping --------------------------------------------------------
    def event(self, **kw) -> None:
        kw["t"] = time.strftime("%H:%M:%S")
        line = json.dumps(kw)
        with self.log_lock:
            print(line, flush=True)
            with open(self.out / "progress.jsonl", "a") as f:
                f.write(line + "\n")

    def run_dir(self, scen_slug: str, name: str) -> Path:
        return self.out / "runs" / scen_slug / name

    # -- one run ------------------------------------------------------------
    def run(self, index: int, scen_slug: str, name: str, *, reference: str | None,
            set_dir: Path | None = None, only: list[str] | None = None,
            exclude: list[str] | None = None) -> dict:
        scenario = self.scenarios[index]
        d = self.run_dir(scen_slug, name)
        if (d / "result.json").is_file():  # resume
            res = json.loads((d / "result.json").read_text())
            res["_dir"] = str(d)
            return res
        d.mkdir(parents=True, exist_ok=True)
        a = self.args
        spec = {"scenarios": a.scenarios, "index": index, "candidates": set_dir is not None,
                "result": str(d / "result.json"),
                "run": {"out_dir": str(d), "seed": a.seed, "max_requests": a.max_requests,
                        "correctness_samples": a.correctness_samples, "enforce_eager": a.eager,
                        "reference": reference, "workloads": a.workloads.split(",") if a.workloads else None}}
        (d / "spec.json").write_text(json.dumps(spec, indent=1))
        env = {k: v for k, v in os.environ.items() if not k.startswith("FASTKERNELS_CANDIDATE")}
        if set_dir is not None:
            env.update(FASTKERNELS_CANDIDATE_DIR=str(set_dir),
                       FASTKERNELS_CANDIDATE_ONLY=",".join(only or []),
                       FASTKERNELS_CANDIDATE_EXCLUDE=",".join(exclude or []),
                       TORCH_EXTENSIONS_DIR=str(self.out / "work" / "torch_extensions" / set_dir.name))
        gpus, port = self.pool.lease(max(1, scenario.tp))
        env.update(CUDA_VISIBLE_DEVICES=",".join(gpus), FASTKERNELS_NCCL_PORT=str(port))
        t0 = time.time()
        timed_out = False
        try:
            with open(d / "log.txt", "w") as log:
                proc = subprocess.Popen([sys.executable, "-m", "fastkernels.e2e.runner", str(d / "spec.json")],
                                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    proc.wait(timeout=a.run_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        finally:
            self.pool.release(gpus)
        if (d / "result.json").is_file():
            res = json.loads((d / "result.json").read_text())
        else:
            res = {"status": "crash", "error": ("TIMEOUT" if timed_out else "no result (process died)")}
            (d / "result.json").write_text(json.dumps(res, indent=1))
        res["_dir"] = str(d)
        self.event(scenario=scen_slug, run=name, gpus=gpus, status=res["status"],
                   wall_s=round(time.time() - t0), error=(res.get("error") or "")[:200])
        return res

    # -- scoring ------------------------------------------------------------
    def score(self, index: int, base: dict, res: dict) -> dict:
        """Speedups vs baseline and adapter correctness vs baseline outputs."""
        import torch
        from .adapters import adapter_for
        out: dict = {"speedups": {}}
        bt = base.get("timings") or {}
        for wl, t in (res.get("timings") or {}).items():
            b = bt.get(wl)
            if b and b.get("value") and t.get("value"):
                out["speedups"][wl] = (t["value"] / b["value"] if t["kind"] == "throughput"
                                       else b["value"] / t["value"])
        ref = Path(base["_dir"]) / "outputs.pt"
        cand = Path(res["_dir"]) / "outputs.pt"
        if ref.is_file() and cand.is_file():
            try:
                cmp = adapter_for(self.scenarios[index]).compare(
                    torch.load(ref, weights_only=False), torch.load(cand, weights_only=False))
                out["correctness"] = cmp
                ps = cmp.get("per_sample") or []
                out["mean_d"] = sum(ps) / len(ps) if ps else None
            except Exception as exc:  # noqa: BLE001
                out["correctness_error"] = f"{type(exc).__name__}: {exc}"
        return out

    # -- chains -------------------------------------------------------------
    def scenario_chain(self, index: int) -> None:
        scenario = self.scenarios[index]
        slug = f"{index:02d}_{_slug(scenario.hf_name)}"
        res_dir = self.out / "results" / slug
        res_dir.mkdir(parents=True, exist_ok=True)
        base = self.run(index, slug, "baseline", reference=None)
        (res_dir / "baseline.json").write_text(json.dumps(
            {"model": scenario.hf_name, "tp": scenario.tp, **{k: v for k, v in base.items() if k != "traceback"}},
            indent=1))
        if base["status"] != "ok":
            for s in self.sets:
                (res_dir / f"{s.name}.json").write_text(json.dumps(
                    {"model": scenario.hf_name, "set": s.name, "status": "baseline_failed",
                     "error": base.get("error")}, indent=1))
            return
        ref = str(Path(base["_dir"]) / "outputs.pt")
        with ThreadPoolExecutor(max_workers=1 + len(self.sets)) as ex:
            futs = [] if self.args.skip_noise else [ex.submit(self.noise_chain, index, slug, base, ref)]
            futs += [ex.submit(self.set_chain, index, slug, base, ref, s) for s in self.sets]
            for f in futs:
                f.result()

    def noise_chain(self, index: int, slug: str, base: dict, ref: str) -> None:
        path = self.out / "results" / slug / "noise.json"
        if path.is_file():
            return
        res = self.run(index, slug, "noise", reference=ref)
        summary = {"model": self.scenarios[index].hf_name, "status": res["status"],
                   "error": res.get("error"), "timings": res.get("timings")}
        if res["status"] == "ok":
            summary.update(self.score(index, base, res))
        path.write_text(json.dumps(summary, indent=1))

    def set_chain(self, index: int, slug: str, base: dict, ref: str, set_dir: Path) -> None:
        path = self.out / "results" / slug / f"{set_dir.name}.json"
        if path.is_file():
            return
        try:
            summary = self._drop_and_retry(index, slug, base, ref, set_dir)
        except Exception:  # noqa: BLE001 -- orchestration bug: record, keep other chains going
            summary = {"status": "orchestrator_error", "traceback": traceback.format_exc()}
        summary.update(model=self.scenarios[index].hf_name, set=set_dir.name)
        path.write_text(json.dumps(summary, indent=1))

    def _drop_and_retry(self, index, slug, base, ref, set_dir) -> dict:
        a = self.args
        requested = kernels_for(set_dir, self.scenarios[index].hf_name)
        present = {f"{p.parent.name}:{p.stem}" for p in set_dir.glob("L[1-4]/*.py")}
        requested = [k for k in requested if k in present]
        dropped: list[dict] = []
        attempts: list[dict] = []
        n = 0

        def attempt(only: list[str], purpose: str) -> tuple[dict, dict]:
            nonlocal n
            n += 1
            excl = [x["kernel"] for x in dropped]
            res = self.run(index, slug, f"{set_dir.name}/attempt_{n:02d}", reference=ref,
                           set_dir=set_dir, only=only, exclude=excl)
            log = (Path(res["_dir"]) / "log.txt").read_text(errors="replace") \
                if (Path(res["_dir"]) / "log.txt").is_file() else ""
            info = {"n": n, "purpose": purpose, "only": only, "exclude": excl, "status": res["status"],
                    "error": (res.get("error") or "")[:1500], "swapped": res.get("swapped"),
                    "not_swapped": res.get("not_swapped"), "t_import_s": res.get("t_import_s"),
                    "t_total_s": res.get("t_total_s")}
            if res["status"] == "ok":
                sc = self.score(index, base, res)
                info.update(speedups=sc["speedups"], mean_d=sc.get("mean_d"))
                res["_score"] = sc
            else:
                info["category"] = _categorize(res.get("error") or "", log)
                info["culprits"] = culprits(log + "\n" + (res.get("traceback") or ""), set_dir)
            attempts.append(info)
            return res, info

        def broken(info: dict) -> bool:
            return info["status"] == "ok" and info.get("mean_d") is not None \
                and info["mean_d"] > a.broken_threshold

        def bisect(active: list[str], bad) -> str | None:
            """Find one kernel whose presence alone makes ``bad`` true (else None)."""
            suspects = list(active)
            while len(suspects) > 1 and n < a.max_attempts:
                half = suspects[: len(suspects) // 2]
                _, info = attempt(half, "bisect")
                if bad(info):
                    suspects = half
                    continue
                rest = suspects[len(half):]
                if n >= a.max_attempts:
                    return None
                _, info = attempt(rest, "bisect")
                if bad(info):
                    suspects = rest
                    continue
                return None  # interaction between halves: give up bisecting
            return suspects[0] if len(suspects) == 1 else None

        final = None
        while n < a.max_attempts:
            active = [k for k in requested if k not in {x["kernel"] for x in dropped}]
            if not active:
                break
            res, info = attempt(active, "full")
            for k in info.get("not_swapped") or []:
                if k not in {x["kernel"] for x in dropped}:
                    dropped.append({"kernel": k, "category": "import", "attempt": info["n"],
                                    "reason": "candidate failed to import / define its class"})
            if info["status"] == "ok" and not broken(info):
                final = (res, info)
                break
            if info["status"] != "ok":
                named = [k for k in info["culprits"] if k not in {x["kernel"] for x in dropped}]
                if named:
                    dropped.append({"kernel": named[0], "category": info["category"],
                                    "attempt": info["n"], "reason": info["error"][:500]})
                    continue
                culprit = bisect(active, lambda i: i["status"] != "ok")
                category = info["category"]
            else:
                culprit = bisect(active, broken)
                category = "incorrect"
            if culprit is None:
                break
            dropped.append({"kernel": culprit, "category": category, "attempt": n,
                            "reason": f"found by bisection ({category})"})

        summary = {"kernels_requested": requested, "dropped": dropped, "attempts": attempts,
                   "all_winners": attempts[0] if attempts else None}
        if final is not None:
            res, info = final
            summary.update(status="ok", kernels_final=info["only"], timings=res.get("timings"),
                           speedups=res["_score"]["speedups"], correctness=res["_score"].get("correctness"),
                           mean_d=res["_score"].get("mean_d"))
        elif not [k for k in requested if k not in {x["kernel"] for x in dropped}]:
            summary.update(status="empty", kernels_final=[],
                           note="no deployable kernels: result equals the baseline")
        else:
            summary.update(status="incomplete", kernels_final=None,
                           note=f"stopped after {n} attempts without a deployable subset")
        return summary

    def main(self) -> int:
        idx = ([int(i) for i in self.args.scenario_indices.split(",")] if self.args.scenario_indices
               else list(range(len(self.scenarios))))
        (self.out / "results").mkdir(parents=True, exist_ok=True)
        self.event(event="start", scenarios=idx, sets=[s.name for s in self.sets], gpus=self.pool.total)
        # One thread per scenario; GPU leasing inside ``run`` does the actual packing.
        with ThreadPoolExecutor(max_workers=max(1, len(idx))) as ex:
            for f in [ex.submit(self.scenario_chain, i) for i in idx]:
                f.result()
        self.event(event="done")
        return 0


def _detect_gpus() -> list[str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [g for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g]
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return ["0"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fastkernels e2e", description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios", help="scenarios table (path or packaged name, e.g. default)")
    ap.add_argument("--sets", required=True, help="comma-separated candidate-set directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--scenario-indices", default=None, help="subset of scenario indices, e.g. 0,4")
    ap.add_argument("--workloads", default=None, help="subset of workloads (smoke tests only)")
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--correctness-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-attempts", type=int, default=16)
    ap.add_argument("--broken-threshold", type=float, default=0.5,
                    help="mean per-sample discrepancy above which a running candidate counts as broken")
    ap.add_argument("--run-timeout", type=int, default=5400)
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--eager", action="store_true", help="diagnostics only: disable torch.compile/graphs")
    args = ap.parse_args(argv)
    return E2E(args).main()


if __name__ == "__main__":
    raise SystemExit(main())
