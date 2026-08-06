#!/usr/bin/env python3
"""Kernel/host breakdown of one Qwen3-Next prefill step and one decode chunk.

Answers three questions the wall-clock grid in ``profile_qwen3_next_gap.py``
cannot:

  1. Where does the ~130 ms fixed cost of a single prefill step go -- host
     (Python metadata construction, pinned allocations, syncs) or device?
  2. Which kernels dominate a decode graph replay, and how much of the step is
     spent outside the graph?
  3. How much of the step is GPU-idle (host-bound) versus GPU-busy?

Runs against rank 0 in-process; the TP peer is a separate process and is not
profiled, which is fine because both ranks run the same shapes.

Usage:
    python tests/debug/profile_qwen3_next_step.py --tp 2 --prefill-len 16384
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


def _fmt_us(us: float) -> str:
    return f"{us / 1000:9.3f}ms"


def summarize(prof, label: str, top: int = 22) -> dict:
    """Aggregate a profile into device/host totals plus the top device kernels."""
    ka = prof.key_averages()
    dev = defaultdict(float)
    host = defaultdict(float)
    dev_total = 0.0
    host_total = 0.0
    for e in ka:
        d = getattr(e, "self_device_time_total", 0.0) or 0.0
        h = getattr(e, "self_cpu_time_total", 0.0) or 0.0
        if d:
            dev[e.key] += d
            dev_total += d
        if h:
            host[e.key] += h
            host_total += h

    print(f"\n  === {label} ===")
    print(f"  device busy total : {_fmt_us(dev_total)}")
    print(f"  host  busy total : {_fmt_us(host_total)}")
    print(f"\n  {'top device kernels':<62} {'self dev':>11}  {'calls':>7}")
    counts = {e.key: e.count for e in ka}
    for k, v in sorted(dev.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {k[:62]:<62} {_fmt_us(v):>11}  {counts.get(k, 0):>7}")
    print(f"\n  {'top host ops (self)':<62} {'self host':>11}  {'calls':>7}")
    for k, v in sorted(host.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {k[:62]:<62} {_fmt_us(v):>11}  {counts.get(k, 0):>7}")
    return {
        "device_total_us": dev_total,
        "host_total_us": host_total,
        "device": dict(sorted(dev.items(), key=lambda kv: -kv[1])[:60]),
        "host": dict(sorted(host.items(), key=lambda kv: -kv[1])[:60]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--prefill-len", type=int, default=16384)
    ap.add_argument("--decode-bs", type=int, default=1)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--decode-prompt-len", type=int, default=256,
                    help="context length each decode sequence carries")
    ap.add_argument("--decode-skew-len", type=int, default=0,
                    help="if set, give sequence 0 this context instead, so the "
                         "batch has one long sequence among short ones (the "
                         "long-context workload's shape)")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--out", default="/tmp/qwen3_next_step_profile.json")
    args = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from fastkernels.infra.engine import LlamaEngine, SamplingParams, Sequence

    engine = LlamaEngine(
        model_name=MODEL, seed=42, enforce_eager=False,
        tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
    )
    mr = engine.model_runner
    engine.generate([[0] * 16],
                    SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))

    rng = random.Random(7)
    out = {}

    # ---------------------------------------------------------------- prefill
    # Drive mr.call("run", ...) directly so the measurement covers exactly one
    # prefill forward step, with none of the generate() scheduling around it.
    def fresh_prefill_seq(n_tok):
        mr.call("reset_kimi_state_cache")
        seq = Sequence([rng.randrange(1000, 100000) for _ in range(n_tok)],
                       max_tokens=1, ignore_eos=True)
        slots = mr.call("allocate_mamba_state_batch", 1)
        seq.state_slot = slots[0]
        return seq

    n_tok = args.prefill_len
    for _ in range(2):
        seq = fresh_prefill_seq(n_tok)
        mr.call("run", [seq], True, [n_tok])
    torch.cuda.synchronize()

    reps = 3
    seqs = [fresh_prefill_seq(n_tok) for _ in range(reps)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in seqs:
        mr.call("run", [s], True, [n_tok])
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / reps
    print(f"\n  prefill step wall (bs=1, {n_tok} tok): {wall * 1e3:.2f}ms", flush=True)
    out["prefill_wall_ms"] = wall * 1e3

    seq = fresh_prefill_seq(n_tok)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        mr.call("run", [seq], True, [n_tok])
        torch.cuda.synchronize()
    out["prefill"] = summarize(prof, f"prefill bs=1 x {n_tok} tok "
                                     f"(wall {wall * 1e3:.1f}ms)")

    # ------------------------------------------------- engine-loop overhead
    # ``generate(max_tokens=1)`` is what the latency benchmark measures, so the
    # difference against the bare forward above is everything the scheduler adds
    # per request: the state-cache reset, admission, slot claim + zero-fill, the
    # SHM round trips and sampling. Worth separating before attributing a
    # single-request gap to the model.
    prompt = [rng.randrange(1000, 100000) for _ in range(n_tok)]
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    for _ in range(2):
        engine.block_manager.reset()
        engine.generate([prompt], sp1, use_tqdm=False, decode_text=False)
    gen = []
    for _ in range(reps):
        engine.block_manager.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.generate([prompt], sp1, use_tqdm=False, decode_text=False)
        torch.cuda.synchronize()
        gen.append(time.perf_counter() - t0)
    gen_wall = min(gen)
    out["generate_wall_ms"] = gen_wall * 1e3
    print(f"\n  generate(max_tokens=1) bs=1 x {n_tok} tok: {gen_wall * 1e3:.2f}ms"
          f"   (forward {wall * 1e3:.2f}ms, engine loop "
          f"{(gen_wall - wall) * 1e3:.2f}ms)", flush=True)

    # ----------------------------------------------------------- decode chunk
    bs = args.decode_bs
    steps = args.decode_steps
    plen = args.decode_prompt_len
    dseqs = []
    mr.call("reset_kimi_state_cache")
    slots = mr.call("allocate_mamba_state_batch", bs)
    for i in range(bs):
        n = args.decode_skew_len if (i == 0 and args.decode_skew_len) else plen
        s = Sequence([rng.randrange(1000, 100000) for _ in range(n)],
                     max_tokens=steps + 8, ignore_eos=True)
        s.state_slot = slots[i]
        dseqs.append(s)
    # Prefill each sequence in chunks the scheduler would use, so the KV cache
    # and GDN state hold real content at the right positions.
    budget = mr.max_num_batched_tokens
    pending = list(dseqs)
    while pending:
        batch, lens, left = [], [], budget
        for s in pending:
            rem = len(s.token_ids) - s.num_computed_tokens
            if rem <= 0:
                continue
            take = min(rem, left)
            if take <= 0:
                break
            batch.append(s)
            lens.append(take)
            left -= take
            if left <= 0:
                break
        if not batch:
            break
        mr.call("run", batch, True, lens)
        pending = [s for s in pending
                   if len(s.token_ids) - s.num_computed_tokens > 0]
    for s in dseqs:
        s.append_token(rng.randrange(1000, 100000))
    ctx = sorted(len(s.token_ids) for s in dseqs)
    print(f"\n  decode contexts: min={ctx[0]:,} max={ctx[-1]:,}", flush=True)

    mr.call("run_kimi_decode_many", dseqs, 4)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mr.call("run_kimi_decode_many", dseqs, steps)
    torch.cuda.synchronize()
    dwall = (time.perf_counter() - t0) / steps
    print(f"\n  decode step wall (bs={bs}): {dwall * 1e3:.3f}ms", flush=True)
    out["decode_wall_ms"] = dwall * 1e3

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        mr.call("run_kimi_decode_many", dseqs, steps)
        torch.cuda.synchronize()
    out["decode"] = summarize(
        prof, f"decode bs={bs} x {steps} steps (wall {dwall * 1e3:.3f}ms/step)")
    out["decode_steps"] = steps
    out["decode_bs"] = bs

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
