#!/usr/bin/env python3
"""Split the fastkernels-vs-vLLM gap on Qwen3-Next into prefill and decode.

Runs a grid of (batch_size, input_len, output_len) points twice per engine:
once with ``max_tokens=1`` (prefill only) and once with the full output length.
Decode time is the difference, so every number is measured through the same
public ``generate()`` entry point both benchmarks use -- no instrumentation
inside the engines, and identical synthetic prompts on both sides.

Usage:
    python tests/debug/profile_qwen3_next_gap.py --engine fastkernels --tp 2
    python tests/debug/profile_qwen3_next_gap.py --engine vllm --tp 2

Both write JSON to --out (default /tmp/qwen3_next_gap_<engine>.json) so the
two runs can be diffed afterwards with --compare.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# (name, batch_size, input_len, output_len)
GRID = [
    ("bs1-in596",     1,   596, 128),
    ("bs1-in4k",      1,  4096, 128),
    ("bs32-in128",   32,   128, 128),
    ("bs32-in3812",  32,  3812, 128),
    ("bs64-in128",   64,   128, 128),
    ("bs256-in128", 256,   128, 128),
    # Prefill-shape probes (output_len 1 => prefill only, decode row is ~0).
    ("pf-1x16k",      1, 16384,   1),
    ("pf-1x32k",      1, 32768,   1),
    ("pf-4x16k",      4, 16384,   1),
    ("pf-8x8k",       8,  8192,   1),
    # Long-context probes: the LongBench-v2 workload runs 8K..128K prompts, and
    # the full-attention layers are quadratic in position, so the per-chunk cost
    # keeps climbing well past 32K.
    ("pf-1x64k",      1, 65536,   1),
    ("pf-1x128k",     1, 131072,  1),
    ("pf-2x64k",      2, 65536,   1),
]


def make_prompts(bs: int, in_len: int, seed: int = 1234) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(1000, 100000) for _ in range(in_len)]
        for _ in range(bs)
    ]


def _stats(samples: list[float]) -> dict:
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# fastkernels
# ---------------------------------------------------------------------------
def run_fastkernels(args) -> dict:
    import torch
    from fastkernels.infra.engine import LlamaEngine, SamplingParams

    engine = LlamaEngine(
        model_name=MODEL,
        seed=42,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
    )
    engine.generate(
        [[0] * 16],
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
    )

    def timed(prompts, max_tokens, iters):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            ignore_eos=True)
        for _ in range(args.warmup):
            engine.block_manager.reset()
            engine.generate(prompts, sp, use_tqdm=False, decode_text=False)
        out = []
        for _ in range(iters):
            engine.block_manager.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(prompts, sp, use_tqdm=False, decode_text=False)
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
        return out

    return _run_grid(args, timed)


# ---------------------------------------------------------------------------
# vLLM (in-process; tp>1 uses the mp executor like bench_vllm does)
# ---------------------------------------------------------------------------
def run_vllm(args) -> dict:
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    import torch
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        seed=42,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
    )
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    llm = LLM(**kwargs)
    llm.generate(
        [dict(prompt_token_ids=[0] * 16)],
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
    )

    def timed(prompts, max_tokens, iters):
        vp = [dict(prompt_token_ids=p) for p in prompts]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            ignore_eos=True, detokenize=False)
        for _ in range(args.warmup):
            llm.generate(vp, sp, use_tqdm=False)
        out = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(vp, sp, use_tqdm=False)
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
        return out

    return _run_grid(args, timed)


# ---------------------------------------------------------------------------
def _run_grid(args, timed) -> dict:
    results = {}
    grid = [g for g in GRID if not args.only or g[0] in args.only]
    for name, bs, in_len, out_len in grid:
        if in_len + out_len > args.max_model_len:
            continue
        prompts = make_prompts(bs, in_len)
        pf = timed(prompts, 1, args.iters)
        full = timed(prompts, out_len, args.iters) if out_len > 1 else pf
        pf_med = statistics.median(pf)
        full_med = statistics.median(full)
        decode_med = full_med - pf_med
        steps = max(out_len - 1, 0)
        row = {
            "batch_size": bs,
            "input_len": in_len,
            "output_len": out_len,
            "prefill": _stats(pf),
            "full": _stats(full),
            "prefill_s": pf_med,
            "decode_s": decode_med,
            "prefill_tok_per_s": (bs * in_len) / pf_med if pf_med > 0 else 0.0,
            "ms_per_decode_step": (decode_med / steps * 1e3) if steps else 0.0,
        }
        results[name] = row
        print(
            f"  {name:<14} bs={bs:<4} in={in_len:<6} out={out_len:<4} "
            f"prefill={pf_med * 1e3:8.2f}ms ({row['prefill_tok_per_s']:>9,.0f} tok/s) "
            f"decode={decode_med * 1e3:8.2f}ms "
            f"({row['ms_per_decode_step']:6.3f} ms/step)",
            flush=True,
        )
    return results


def compare(a_path: str, b_path: str) -> None:
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    print(f"\n  {'SCENARIO':<14} {'fk prefill':>11} {'vllm prefill':>13} {'ratio':>7}"
          f"   {'fk ms/step':>11} {'vllm ms/step':>13} {'ratio':>7}")
    print("  " + "-" * 88)
    for name in a.get("grid", {}):
        if name not in b.get("grid", {}):
            continue
        ra, rb = a["grid"][name], b["grid"][name]
        pa, pb = ra["prefill_s"] * 1e3, rb["prefill_s"] * 1e3
        da, db = ra["ms_per_decode_step"], rb["ms_per_decode_step"]
        print(f"  {name:<14} {pa:>10.2f}m {pb:>12.2f}m {pa / pb if pb else 0:>7.2f}"
              f"   {da:>10.3f}m {db:>12.3f}m {da / db if db else 0:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fastkernels", "vllm"], default="fastkernels")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    print(f"\n  {args.engine} / {MODEL} / tp={args.tp} / "
          f"eager={args.enforce_eager}\n", flush=True)
    t0 = time.perf_counter()
    grid = (run_fastkernels if args.engine == "fastkernels" else run_vllm)(args)
    out = args.out or f"/tmp/qwen3_next_gap_{args.engine}.json"
    with open(out, "w") as f:
        json.dump({
            "engine": args.engine, "tp": args.tp, "model": MODEL,
            "enforce_eager": args.enforce_eager, "grid": grid,
        }, f, indent=1)
    print(f"\n  wrote {out}  ({time.perf_counter() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
