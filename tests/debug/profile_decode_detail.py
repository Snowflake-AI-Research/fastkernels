#!/usr/bin/env python3
"""Detailed profiling of a single decode step using torch profiler."""

import os
import sys
import time
import torch
from random import randint, seed as set_seed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastkernels.infra.engine import LlamaEngine, SamplingParams

MODEL = os.environ.get("MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
TP = int(os.environ.get("TP", "4"))
BS = int(os.environ.get("BS", "128"))


def main():
    set_seed(0)
    engine = LlamaEngine(model_name=MODEL, tensor_parallel_size=TP, enforce_eager=False)
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    # First generate to prefill all sequences
    ids = [[randint(0, 10000) for _ in range(200)] for _ in range(BS)]
    sp = SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=True)
    engine.generate(ids, sp)

    # Now profile a 20-step decode
    ids2 = [[randint(0, 10000) for _ in range(200)] for _ in range(BS)]
    sp2 = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as prof:
        engine.generate(ids2, sp2)

    print(f"\n=== Top 30 CUDA kernel operations (BS={BS}) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    prof.export_chrome_trace("/tmp/trace_decode.json")
    print("\nTrace exported to /tmp/trace_decode.json")

    del engine


if __name__ == "__main__":
    main()
