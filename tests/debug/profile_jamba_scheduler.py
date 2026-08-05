#!/usr/bin/env python3
"""Account for every forward pass a JambaEngine ``mixed`` run makes.

``mixed`` is the row that degrades with scale -- 0.96x of vLLM at 64 sequences
but 0.83x at 1000 -- which points at the scheduler rather than any kernel. The
suspicion this probe tests: ``JambaEngine.generate`` is phase-pure, so each loop
iteration runs a SEPARATE prefill forward and decode forward. Once the first
``max_num_seqs`` prompts are resident, sequences finish a few per step, so the
engine admits a few, and every iteration then pays a full second pass over all
32 layers' weights to prefill one or two prompts. vLLM's chunked prefill puts
those prompt tokens in the SAME forward as the decode rows.

Prints, for the real WildChat mixed workload:

  * prefill vs decode forward counts and total device time in each,
  * the distribution of prefill widths (a histogram of tokens per prefill call),
  * how many prefill calls were "small" (< 25% of ``max_num_batched_tokens``),
    which is the wasted-pass count the hypothesis predicts.

Usage:
    python tests/debug/profile_jamba_scheduler.py --num-seqs 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from fastkernels.infra.jamba_engine import JambaEngine, SamplingParams  # noqa: E402
from fastkernels.workloads import load_real_prompt_workload  # noqa: E402

MODEL = os.environ.get("MODEL", "ai21labs/AI21-Jamba-Mini-1.7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-seqs", type=int, default=1000)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--floor", type=int, action="append", default=None,
        help="Sweep JambaEngine.prefill_batch_floor over these values in one "
             "process (one model load), reporting wall and the prefill/decode "
             "split for each.",
    )
    ap.add_argument(
        "--defer", type=int, action="append", default=None,
        help="Sweep JambaEngine.prefill_max_defer_steps too (cross product "
             "with --floor).",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    samples = load_real_prompt_workload(
        "mixed", tok, num_requests=args.num_seqs,
        decode_cap=args.max_output_tokens, seed=args.seed,
    )
    prompts = [s.prompt_token_ids[-args.max_prompt_tokens:] for s in samples]
    out_lens = [min(s.output_len, args.max_output_tokens) for s in samples]

    engine = JambaEngine(model_name=MODEL, max_num_seqs=args.max_num_seqs)
    engine.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=4, ignore_eos=True))

    stats = {
        "prefill_calls": 0, "prefill_tokens": 0, "prefill_s": 0.0,
        "decode_calls": 0, "decode_rows": 0, "decode_s": 0.0,
        "prefill_widths": [], "decode_widths": [],
    }
    raw_prefill = engine._run_prefill_chunks
    raw_decode = engine._run_decode_step

    def timed_prefill(chunks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = raw_prefill(chunks)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        width = sum(c for _, c in chunks)
        stats["prefill_calls"] += 1
        stats["prefill_tokens"] += width
        stats["prefill_s"] += dt
        stats["prefill_widths"].append(width)
        return out

    def timed_decode(running):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = raw_decode(running)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        stats["decode_calls"] += 1
        stats["decode_rows"] += len(running)
        stats["decode_s"] += dt
        stats["decode_widths"].append(len(running))
        return out

    engine._run_prefill_chunks = timed_prefill
    engine._run_decode_step = timed_decode

    sp = [SamplingParams(temperature=0.0, max_tokens=ol, ignore_eos=True)
          for ol in out_lens]
    # Same warmup the bench does, so the timed run is not paying JIT.
    engine.generate(prompts, SamplingParams(max_tokens=1, ignore_eos=True))

    for floor in (args.floor or [engine.prefill_batch_floor]):
        for defer in (args.defer or [engine.prefill_max_defer_steps]):
            engine.prefill_batch_floor = floor
            engine.prefill_max_defer_steps = defer
            for k in stats:
                stats[k] = [] if isinstance(stats[k], list) else type(stats[k])(0)

            t0 = time.perf_counter()
            outs = engine.generate(prompts, sp, use_tqdm=True)
            wall = time.perf_counter() - t0
            _report(engine, stats, outs, wall, floor, defer)

    del engine


def _report(engine, stats, outs, wall, floor, defer):
    total_out = sum(len(o.token_ids) for o in outs)
    pw = np.array(stats["prefill_widths"]) if stats["prefill_widths"] else np.zeros(1)
    dw = np.array(stats["decode_widths"]) if stats["decode_widths"] else np.zeros(1)
    budget = engine.max_num_batched_tokens

    print(f"\n  prefill_batch_floor={floor}  prefill_max_defer_steps={defer}")
    print(f"  wall {wall:.2f}s   {total_out / wall:,.0f} tok/s   "
          f"{total_out:,} output tokens")
    print(f"  prefill: {stats['prefill_calls']:>6} calls  "
          f"{stats['prefill_s']:>7.2f}s ({stats['prefill_s'] / wall * 100:.1f}%)  "
          f"{stats['prefill_tokens']:,} tokens  "
          f"{stats['prefill_s'] / max(stats['prefill_calls'], 1) * 1000:.2f} ms/call")
    print(f"  decode : {stats['decode_calls']:>6} calls  "
          f"{stats['decode_s']:>7.2f}s ({stats['decode_s'] / wall * 100:.1f}%)  "
          f"{stats['decode_rows']:,} rows    "
          f"{stats['decode_s'] / max(stats['decode_calls'], 1) * 1000:.2f} ms/call")
    print(f"  prefill width: median {np.median(pw):.0f}  mean {pw.mean():.0f}  "
          f"max {pw.max():.0f}  (budget {budget})")
    print(f"  decode  width: median {np.median(dw):.0f}  mean {dw.mean():.0f}  "
          f"max {dw.max():.0f}  (cap {engine.max_num_seqs})")

    small = int((pw < budget * 0.25).sum())
    small_s = float(
        sum(w for w in stats["prefill_widths"] if w < budget * 0.25)
    )
    print(f"  prefill calls under 25% of budget: {small}/{len(pw)} "
          f"carrying {small_s:,.0f} of {pw.sum():,.0f} prompt tokens")
    for lo, hi in [(0, 512), (512, 2048), (2048, 8192), (8192, 1 << 30)]:
        n = int(((pw >= lo) & (pw < hi)).sum())
        if n:
            print(f"      width [{lo:>5}, {hi if hi < 1 << 30 else 'inf':>5}): "
                  f"{n:>5} calls")


if __name__ == "__main__":
    main()
