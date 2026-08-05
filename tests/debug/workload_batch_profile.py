#!/usr/bin/env python3
"""Ideal decode-batch profile for a workload, given instantaneous admission.

Total output tokens are fixed, so a throughput run's wall clock is set by how
many *steps* they are spread over -- i.e. by occupancy. The floor is what you get
if every request is admitted at step 0 and never preempted: batch at step ``t``
is ``#{i : out_len_i > t}``, the step count is ``max(out_len)``, and the mean
batch is ``sum(out_len) / max(out_len)``.

Comparing a measured histogram against this says whether a low mean batch is the
workload's own shape (a long tail of short requests) or the engine's admission.

Usage: python tests/debug/workload_batch_profile.py [--scenario mixed] [--cap 1024]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--scenario", default="mixed")
    ap.add_argument("--num-seqs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=1024,
                    help="max_num_seqs, for the band edges")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=0,
                    help="KV pool size in blocks (the engine prints it as "
                         "'KV cache: N blocks'), to compare against demand")
    args = ap.parse_args()

    import numpy as np
    from transformers import AutoTokenizer

    from fastkernels.workloads import (
        DEFAULT_DECODE_CAPS, DEFAULT_WORKLOAD_DATASETS,
        load_real_prompt_workload,
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    samples = load_real_prompt_workload(
        args.scenario, tok, num_requests=args.num_seqs,
        decode_cap=DEFAULT_DECODE_CAPS[args.scenario],
        dataset_name=DEFAULT_WORKLOAD_DATASETS[args.scenario],
        seed=args.seed,
    )
    out = np.array([s.output_len for s in samples])
    inp = np.array([len(s.prompt_token_ids) for s in samples])

    print(f"\n  {len(out)} requests")
    print(f"  prompt len : min={inp.min()} p50={int(np.percentile(inp, 50))} "
          f"p90={int(np.percentile(inp, 90))} max={inp.max()} sum={inp.sum():,}")
    print(f"  output len : min={out.min()} p50={int(np.percentile(out, 50))} "
          f"p90={int(np.percentile(out, 90))} max={out.max()} sum={out.sum():,}")

    steps = int(out.max())
    batch = np.array([(out > t).sum() for t in range(steps)])
    print("\n  IDEAL (all admitted at step 0, no preemption)")
    print(f"    decode steps      {steps}")
    print(f"    mean decode batch {batch.mean():.0f}")
    bands = " ".join(
        f"{i * args.cap // 8}-{(i + 1) * args.cap // 8}:"
        f"{int(((batch * 8) // args.cap == i).sum())}"
        for i in range(8)
        if ((batch * 8) // args.cap == i).sum()
    )
    print(f"    bands             {bands}")
    for frac in (0.25, 0.5, 0.75):
        n_below = int((batch < frac * batch.max()).sum())
        print(f"    steps below {frac:.0%} of peak batch: {n_below}")

    # Peak KV demand under the same instantaneous-admission model. This is the
    # quantity the admission gate actually needs: request i holds
    # ceil((prompt_i + t)/block) blocks while it is live, and requests retire
    # while others grow, so the peak is far below sum(final length) whenever
    # output lengths are heterogeneous. Comparing against ``--blocks`` says
    # whether the pool is genuinely oversubscribed.
    b = args.block_size
    held = np.array([
        int(((inp[out > t] + t + b - 1) // b).sum()) for t in range(steps)
    ])
    final = int((((inp + out) + b - 1) // b).sum())
    print(f"\n  KV DEMAND (block_size={b})")
    print(f"    sum of final lengths   {final:,} blocks")
    print(f"    true peak in flight    {held.max():,} blocks "
          f"(at step {int(held.argmax())})")
    print(f"    peak / sum-of-final    {held.max() / final:.2f}x")
    if args.blocks:
        print(f"    pool                   {args.blocks:,} blocks -> "
              f"sum-of-final is {final / args.blocks:.2f}x the pool, "
              f"true peak is {held.max() / args.blocks:.2f}x")


if __name__ == "__main__":
    main()
