#!/usr/bin/env python3
"""Profile Llama TP=1 decode step latency at batch size 256."""

import time
import torch
from random import randint, seed as set_seed
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastkernels.infra.engine import LlamaEngine, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TP = 1


def main():
    set_seed(0)

    engine = LlamaEngine(model_name=MODEL, tensor_parallel_size=TP, enforce_eager=False)
    engine.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=16))

    for bs in [128, 256, 384, 512]:
        set_seed(42)
        ids = [[randint(0, 10000) for _ in range(100)] for _ in range(bs)]
        sp = SamplingParams(temperature=0.0, max_tokens=50, ignore_eos=True)

        engine.generate(ids, sp)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = engine.generate(ids, sp)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total = sum(len(o.token_ids) for o in outputs)
        ms_per_step = elapsed / 50 * 1000
        print(f"  BS={bs:>3d}: {elapsed:.3f}s, {total/elapsed:>8,.0f} tok/s, {ms_per_step:.1f}ms/step")

    del engine


if __name__ == "__main__":
    main()
