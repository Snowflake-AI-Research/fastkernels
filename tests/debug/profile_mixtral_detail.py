#!/usr/bin/env python3
"""Detailed CUDA profiling of Mixtral TP=4 decode steps at BS=128."""

import time
import torch
from random import randint, seed as set_seed
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kb_nano.infra.engine import LlamaEngine, SamplingParams

MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TP = 4
BS = 128


def main():
    set_seed(0)
    engine = LlamaEngine(model_name=MODEL, tensor_parallel_size=TP, enforce_eager=False)
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    set_seed(42)
    ids = [[randint(0, 10000) for _ in range(100)] for _ in range(BS)]
    sp = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True)

    engine.generate(ids, sp)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        engine.generate(ids, sp)

    print("\n=== Top CUDA Kernels (by total time) ===")
    table = prof.key_averages()
    table = sorted(table, key=lambda x: -x.device_time_total)
    total_cuda = sum(e.device_time_total for e in table if e.device_time_total > 0)
    cum = 0
    for e in table[:30]:
        if e.device_time_total <= 0:
            continue
        cum += e.device_time_total
        pct = e.device_time_total / total_cuda * 100
        cum_pct = cum / total_cuda * 100
        print(f"  {pct:5.1f}% ({cum_pct:5.1f}% cum) {e.count:>4d}x  {e.device_time_total/1000:>8.1f}ms  {e.key}")
    print(f"\n  Total CUDA time: {total_cuda/1000:.1f}ms")

    del engine


if __name__ == "__main__":
    main()
