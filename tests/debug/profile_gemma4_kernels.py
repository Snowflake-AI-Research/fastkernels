#!/usr/bin/env python3
"""Kernel census of Gemma4 decode and prefill steps, for either engine.

``profile_gemma4_phases.py`` localises the mixed-scenario gap to *both* phases
(prefill 1.45x slower, decode 1.12x slower). This attributes each phase to
kernels.

Decode is measured by *slope*, not by differencing against a ``max_tokens=1``
run: chunked prefill interleaves decode steps into the prefill phase, so a
1-vs-N diff charges those interleaved decode steps to the N-step decode census
and inflates its launch counts. Profiling ``steps_a`` and ``steps_b`` instead
makes the whole prefill phase (and its interleaved decodes) cancel exactly,
leaving ``steps_b - steps_a`` pure decode steps.

Reporting device time *and* wall time per step separates two different faults:
a kernel that is simply slower, versus a step that is the same on the GPU but
spends longer idle waiting on the host.

CAVEAT on the wall/idle numbers: torch.profiler adds host overhead per kernel
launch, and these steps launch ~700 kernels each, so the "idle" line is inflated
by profiling and grows with ``steps_b``. Trust the *device* census here and take
wall-clock from ``profile_gemma4_phases.py`` (unprofiled) instead. Concretely:
this script once reported 39.6% decode idle where the unprofiled run showed
decode fully GPU-bound. Idle is meaningful for the single prefill step, where
launches are few relative to the work.

Usage:
    python tests/debug/profile_gemma4_kernels.py --engine fastkernels --decode-bs 256
    python tests/debug/profile_gemma4_kernels.py --engine vllm --decode-bs 256
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

MODEL = "google/gemma-4-26B-A4B-it"
TRACE_DIR = "/tmp/gemma4_trace"


def _census(events) -> tuple[dict, dict]:
    dev, cnt = defaultdict(float), defaultdict(int)
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dev[e["name"]] += e.get("dur", 0)
        cnt[e["name"]] += 1
    return dev, cnt


def _load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f).get("traceEvents", [])


def _diff_report(label, dev_a, cnt_a, dev_b, cnt_b, steps, wall_s, topn=24):
    """Report (b - a) as the per-step cost of ``steps`` decode steps."""
    dev = {k: dev_b.get(k, 0.0) - dev_a.get(k, 0.0) for k in set(dev_b) | set(dev_a)}
    cnt = {k: cnt_b.get(k, 0) - cnt_a.get(k, 0) for k in set(cnt_b) | set(cnt_a)}
    dev = {k: v for k, v in dev.items() if v > 0}
    total_us = sum(dev.values())
    total_n = sum(max(0, cnt.get(k, 0)) for k in dev)
    print(f"\n  === {label} ({steps} steps) ===")
    print(f"  device kernel time : {total_us / 1000:9.3f}ms "
          f"=> {total_us / 1000 / steps:7.3f} ms/step")
    print(f"  wall time          : {wall_s * 1e3:9.3f}ms "
          f"=> {wall_s * 1e3 / steps:7.3f} ms/step")
    idle = wall_s * 1e3 - total_us / 1000
    print(f"  idle / host bound  : {idle:9.3f}ms "
          f"=> {idle / steps:7.3f} ms/step  ({idle / (wall_s * 1e3) * 100:.1f}% of wall)")
    print(f"  kernel launches    : {total_n:9d} => {total_n / steps:7.1f}/step")
    print(f"\n  {'top device kernels':<58} {'self dev':>11} {'calls':>8} {'per step':>9}")
    for k, v in sorted(dev.items(), key=lambda kv: -kv[1])[:topn]:
        n = max(0, cnt.get(k, 0))
        print(f"  {k[:58]:<58} {v / 1000:>9.3f}ms {n:>8} {n / steps:>9.1f}")
    return {"device_ms": total_us / 1000, "wall_ms": wall_s * 1e3,
            "launches": total_n, "steps": steps}


def run_fastkernels(args):
    import torch

    from fastkernels.infra.engine import LlamaEngine, SamplingParams

    prompts = [[13] * args.prompt_len for _ in range(args.decode_bs)]
    max_model_len = args.prompt_len + args.steps_b + 16
    engine = LlamaEngine(
        model_name=MODEL, seed=args.seed, enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp, max_model_len=max_model_len,
    )

    def run(max_tokens, profile):
        sp = SamplingParams(temperature=0.0, ignore_eos=True,
                            max_tokens=max_tokens)
        torch.cuda.synchronize()
        if not profile:
            engine.generate(prompts, sp, use_tqdm=False, decode_text=False)
            torch.cuda.synchronize()
            return None, 0.0
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False,
        )
        with prof:
            t0 = time.perf_counter()
            engine.generate(prompts, sp, use_tqdm=False, decode_text=False)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        path = f"{TRACE_DIR}/fk_{max_tokens}.json"
        prof.export_chrome_trace(path)
        return _load_trace(path), wall

    run(4, False)  # warm every shape the timed runs will touch
    run(args.steps_b + 1, False)
    ev1, w1 = run(1, True)
    evA, wA = run(args.steps_a + 1, True)
    evB, wB = run(args.steps_b + 1, True)
    d1, c1 = _census(ev1)
    dA, cA = _census(evA)
    dB, cB = _census(evB)
    print(f"\n  prefill wall {w1 * 1e3:.3f}ms   "
          f"steps_a({args.steps_a}) wall {wA * 1e3:.3f}ms   "
          f"steps_b({args.steps_b}) wall {wB * 1e3:.3f}ms")
    _diff_report(f"fastkernels prefill bs={args.decode_bs} len={args.prompt_len}",
                 {}, {}, d1, c1, 1, w1)
    return _diff_report(
        f"fastkernels decode bs={args.decode_bs} (slope)", dA, cA, dB, cB,
        args.steps_b - args.steps_a, wB - wA,
    )


def run_vllm(args):
    import torch
    from vllm import LLM, SamplingParams

    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    prompts = [dict(prompt_token_ids=[13] * args.prompt_len)
               for _ in range(args.decode_bs)]
    max_model_len = args.prompt_len + args.steps_b + 16
    kwargs = dict(
        model=MODEL, seed=args.seed, trust_remote_code=True,
        enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9, max_model_len=max_model_len,
        enable_prefix_caching=False, disable_log_stats=True,
        profiler_config={"profiler": "torch",
                         "torch_profiler_dir": TRACE_DIR,
                         "torch_profiler_with_stack": False,
                         "torch_profiler_use_gzip": False},
    )
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    llm = LLM(**kwargs)

    def run(max_tokens, profile, tag):
        sp = SamplingParams(temperature=0.0, ignore_eos=True,
                            detokenize=False, max_tokens=max_tokens)
        torch.cuda.synchronize()
        if not profile:
            llm.generate(prompts, sp, use_tqdm=False)
            return None, 0.0
        for f in glob.glob(f"{TRACE_DIR}/*.pt.trace.json*"):
            os.remove(f)
        llm.start_profile()
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        wall = time.perf_counter() - t0
        llm.stop_profile()
        # stop_profile() returns before the worker has flushed its trace.
        for _ in range(120):
            files = sorted(glob.glob(f"{TRACE_DIR}/*.pt.trace.json*"))
            if files and os.path.getsize(files[0]) > 0:
                prev = -1
                while prev != os.path.getsize(files[0]):
                    prev = os.path.getsize(files[0])
                    time.sleep(0.5)
                break
            time.sleep(0.5)
        files = sorted(glob.glob(f"{TRACE_DIR}/*.pt.trace.json*"))
        if not files:
            raise SystemExit(f"no vLLM trace appeared in {TRACE_DIR}")
        ev = _load_trace(files[0])
        shutil.move(files[0], f"{TRACE_DIR}/vllm_{tag}.json")
        return ev, wall

    run(4, False, "warm")
    run(args.steps_b + 1, False, "warm")
    ev1, w1 = run(1, True, "prefill")
    evA, wA = run(args.steps_a + 1, True, "a")
    evB, wB = run(args.steps_b + 1, True, "b")
    d1, c1 = _census(ev1)
    dA, cA = _census(evA)
    dB, cB = _census(evB)
    print(f"\n  prefill wall {w1 * 1e3:.3f}ms   "
          f"steps_a({args.steps_a}) wall {wA * 1e3:.3f}ms   "
          f"steps_b({args.steps_b}) wall {wB * 1e3:.3f}ms")
    _diff_report(f"vLLM prefill bs={args.decode_bs} len={args.prompt_len}",
                 {}, {}, d1, c1, 1, w1)
    return _diff_report(
        f"vLLM decode bs={args.decode_bs} (slope)", dA, cA, dB, cB,
        args.steps_b - args.steps_a, wB - wA,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fastkernels", "vllm"],
                    default="fastkernels")
    ap.add_argument("--decode-bs", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--steps-a", type=int, default=16,
                    help="Shorter decode run; its prefill phase cancels out.")
    ap.add_argument("--steps-b", type=int, default=80,
                    help="Longer decode run. Census reports steps_b - steps_a.")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(TRACE_DIR, exist_ok=True)
    if args.engine == "vllm":
        run_vllm(args)
    else:
        run_fastkernels(args)


if __name__ == "__main__":
    main()
