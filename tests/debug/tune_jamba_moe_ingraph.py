#!/usr/bin/env python3
"""Tune the Jamba MoE config where it actually runs: inside the decode graph.

The isolated microbenchmark in ``bench_jamba_moe.py`` / ``tune_jamba_moe_config.py``
is not trustworthy for the decode regime. Its pick for M=256
(``BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64``) measured 1.07x faster
standalone but made the real batch-256 decode step 6.7% SLOWER -- 25.89 ms ->
27.62 ms, with ``_fused_moe_kernel`` going 16.12 ms -> 17.91 ms. The same trap
cost this project a day on Mixtral: the engine replays the MoE inside a captured
CUDA graph alongside 32 layers of other work, and that context changes which
config wins.

So this probe times candidates the only way that counts: build the engine, force
a config, re-capture the decode graph for one bucket, and replay it. One model
load covers the whole grid because only the graph is rebuilt per candidate.

Usage:
    FASTKERNELS_JAMBA_BUCKETS=256 python tests/debug/tune_jamba_moe_ingraph.py --bs 256
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

MODEL = os.environ.get("MODEL", "ai21labs/AI21-Jamba-Mini-1.7")

# Candidates around both the heuristic's "large" pick (which wins in-graph at
# M=256) and the standalone tuner's pick (which does not), plus the axes that
# separated them.
CANDIDATES = [
    None,  # whatever get_triton_config resolves -- the current default
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
    # A wider K tile reads 256-byte weight segments instead of 128-byte ones,
    # which is the plausible route to better HBM efficiency on a kernel that is
    # 62% of the step at 5.6 TB/s. It needs fewer pipeline stages to fit the
    # SM's 232 KB of shared memory.
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 5},
    {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
     "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()

    from random import randint, seed as set_seed

    from fastkernels.infra.jamba_engine import JambaEngine, SamplingParams
    from fastkernels.tasks.baseline.L2 import fused_experts as fe_mod
    from profile_jamba_decode import _prefill_to_running  # noqa: E402

    forced: dict = {}
    real_resolver = fe_mod.get_triton_config

    def _resolver(*a, **kw):
        return dict(forced) if forced else real_resolver(*a, **kw)

    fe_mod.get_triton_config = _resolver

    os.environ.setdefault("FASTKERNELS_JAMBA_BUCKETS", str(args.bs))
    engine = JambaEngine(model_name=MODEL, max_num_seqs=args.bs)
    engine.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=4, ignore_eos=True))

    set_seed(1234)
    prompts = [
        [randint(5, 60000) for _ in range(args.prompt_len)] for _ in range(args.bs)
    ]
    running = _prefill_to_running(engine, prompts)

    print(f"  bs={args.bs}  {len(CANDIDATES)} candidates, "
          f"re-capturing the decode graph for each")
    print(f"  {'ms/STEP':>9} {'vs DEFAULT':>11}  CONFIG")
    base = None
    for cfg in CANDIDATES:
        forced.clear()
        if cfg:
            forced.update(cfg)
        # Drop the captured graph so the next step re-records with this config.
        # A fresh mempool per capture: freeing the old graph drops the shared
        # pool's use count to zero, and re-entering it then trips an internal
        # allocator assert.
        engine._decode_graphs.pop(args.bs, None)
        engine._cuda_graph_mempool_id = torch.cuda.graph_pool_handle()
        try:
            with torch.inference_mode():
                engine._capture_decode_graph(args.bs)
                for _ in range(5):
                    engine._run_decode_step(running)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.steps):
                    engine._run_decode_step(running)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / args.steps * 1000
        except Exception as exc:
            # Big BLOCK_SIZE_M x BLOCK_SIZE_N x num_stages combinations exceed
            # the SM's shared memory; skip rather than abandoning the sweep.
            print(f"  {'--':>9} {'--':>11}  {cfg}  ({type(exc).__name__}: "
                  f"{str(exc)[:60]})")
            engine._decode_graphs.pop(args.bs, None)
            continue
        if base is None:
            base = ms
        print(f"  {ms:>9.3f} {base / ms:>10.3f}x  "
              f"{'default (get_triton_config)' if cfg is None else cfg}")

    del engine


if __name__ == "__main__":
    main()
