#!/usr/bin/env python3
"""Kernel census of one vLLM decode step (and one prefill step) for Qwen3-Next.

Counterpart to ``profile_qwen3_next_step.py``. vLLM v1 runs the model in worker
subprocesses, so an in-process torch.profiler sees nothing; instead this uses
vLLM's own profiler hooks (``VLLM_TORCH_PROFILER_DIR`` + ``start_profile``)
and then parses the resulting Chrome trace for device kernels.

The point of comparison is kernels-per-step: at batch 1 a Qwen3-Next decode
step is launch-bound, so step time tracks kernel count almost linearly.

Usage:
    python tests/debug/profile_vllm_qwen3_next_step.py --tp 2
"""

import argparse
import glob
import gzip
import json
import os
import random
import sys
from collections import defaultdict

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
TRACE_DIR = "/tmp/vllm_q3n_trace"


def parse_traces(pattern: str, steps: int, label: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  no trace files matched {pattern}")
        return {}
    # One worker's trace is representative: every rank runs the same shapes.
    path = files[0]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])
    dev = defaultdict(float)
    cnt = defaultdict(int)
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dev[e["name"]] += e.get("dur", 0)
        cnt[e["name"]] += 1
    total_us = sum(dev.values())
    total_n = sum(cnt.values())
    print(f"\n  === vLLM {label} ({os.path.basename(path)}) ===")
    print(f"  device kernel time : {total_us / 1000:.3f}ms over {steps} step(s) "
          f"=> {total_us / 1000 / steps:.3f}ms/step")
    print(f"  kernel launches    : {total_n} => {total_n / steps:.1f}/step")
    print(f"\n  {'top device kernels':<62} {'self dev':>11} {'calls':>8} {'per step':>9}")
    for k, v in sorted(dev.items(), key=lambda kv: -kv[1])[:26]:
        print(f"  {k[:62]:<62} {v / 1000:>9.3f}ms {cnt[k]:>8} {cnt[k] / steps:>9.1f}")
    return {
        "trace": path, "steps": steps,
        "device_us": total_us, "launches": total_n,
        "per_step_ms": total_us / 1000 / steps,
        "launches_per_step": total_n / steps,
        "kernels": {k: [v, cnt[k]] for k, v in
                    sorted(dev.items(), key=lambda kv: -kv[1])[:80]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--prefill-len", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--out", default="/tmp/qwen3_next_vllm_step_profile.json")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    os.makedirs(TRACE_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(TRACE_DIR, "*")):
        os.remove(f)

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, seed=42, trust_remote_code=True, enforce_eager=False,
        tensor_parallel_size=args.tp, gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len, enable_prefix_caching=False,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": TRACE_DIR,
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
        },
    )
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    llm = LLM(**kwargs)

    rng = random.Random(1234)
    short = [rng.randrange(1000, 100000) for _ in range(256)]
    long_p = [rng.randrange(1000, 100000) for _ in range(args.prefill_len)]

    def gen(prompt, max_tokens):
        return llm.generate(
            [dict(prompt_token_ids=prompt)],
            SamplingParams(temperature=0.0, ignore_eos=True,
                           max_tokens=max_tokens, detokenize=False),
            use_tqdm=False,
        )

    # Warm both shapes so no JIT lands inside the profiled window.
    gen(short, args.decode_steps + 1)
    gen(long_p, 1)

    out = {}

    # -------- decode: 1 prefill + N decode steps; prefill is ~1 step of the N+1
    llm.start_profile()
    gen(short, args.decode_steps + 1)
    llm.stop_profile()
    out["decode"] = parse_traces(
        os.path.join(TRACE_DIR, "*.pt.trace.json*"), args.decode_steps,
        f"decode bs=1 ({args.decode_steps} steps + 1 prefill of 256 tok)")

    for f in glob.glob(os.path.join(TRACE_DIR, "*")):
        os.remove(f)

    # -------- prefill only
    llm.start_profile()
    gen(long_p, 1)
    llm.stop_profile()
    out["prefill"] = parse_traces(
        os.path.join(TRACE_DIR, "*.pt.trace.json*"), 1,
        f"prefill bs=1 x {args.prefill_len} tok (1 step + 1 decode)")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
