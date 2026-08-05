#!/usr/bin/env python3
"""Tune the Triton MoE config for Jamba's expert shape (E=16, N=14336) on B200.

``_get_default_config`` keys BLOCK_SIZE_M on the token count M alone, ignoring
how many rows actually land on each expert. For Jamba that is wrong in the
decode regime: at 256 tokens with top-2 of 16 experts each expert sees ~32 rows,
but M=256 selects the "large" heuristic with ``BLOCK_SIZE_M=128``, so
``moe_align_block_size`` pads every expert to 128 rows and three quarters of the
grouped GEMM's work is padding. Profiled in-engine, ``_fused_moe_kernel`` was
16.16 ms of a 26.4 ms batch-256 decode step -- 5.6 TB/s against a 90 GiB weight
sweep, where the same kernel hits 7.2 TB/s at batch 32.

Rather than change the shared heuristic (every MoE model in the repo reads it),
this writes a tuned table to ``L1/moe_configs/``, which ``_get_moe_configs``
searches and which is keyed by ``E`` and ``N`` -- so it applies to Jamba's expert
shape and nothing else.

Measures the whole ``FusedExperts`` op, not one GEMM, so the moe_align/moe_sum
cost of a given BLOCK_SIZE_M is included in the choice.

Usage:
    python tests/debug/tune_jamba_moe_config.py --quick     # spot-check M=256
    python tests/debug/tune_jamba_moe_config.py --write     # full sweep + emit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

NUM_EXPERTS = 16
TOP_K = 2
HIDDEN = 4096
INTERMEDIATE = 14336

# vLLM's tuned tables use this ladder of M keys; ``get_triton_config`` snaps the
# runtime M to the nearest one, so matching the ladder keeps the lookup sane.
M_KEYS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 200, 256, 512, 1024, 1536,
          2048, 3072, 4096]


def _grid(quick: bool):
    if quick:
        return [
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
             "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4}
            for bm in (16, 32, 64, 128)
            for bn in (128, 256)
            for bk in (64, 128)
        ]
    out = []
    for bm in (16, 32, 64, 128):
        for bn in (64, 128, 256):
            for bk in (64, 128):
                for gm in (1, 16, 32):
                    for warps in (4, 8):
                        for stages in (3, 4):
                            out.append({
                                "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                                "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": gm,
                                "num_warps": warps, "num_stages": stages,
                            })
    return out


def _time(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Spot-check a small grid at a few M instead of the "
                         "full sweep.")
    ap.add_argument("--write", action="store_true",
                    help="Write the winning configs to L1/moe_configs/.")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    from fastkernels.tasks.baseline.L1.moe_grouped_gemm import (
        _get_config_file_name,
        _get_default_config,
    )
    from fastkernels.tasks.baseline.L1.topk_softmax import TopKSoftmax
    from fastkernels.tasks.baseline.L2 import fused_experts as fe_mod
    from fastkernels.tasks.baseline.L2.fused_experts import FusedExperts

    # ``FusedExperts`` takes no config argument -- it resolves one internally --
    # so override the resolver for the duration of the sweep rather than adding
    # a tuning-only parameter to the production op.
    forced: dict = {}
    real_resolver = fe_mod.get_triton_config

    def _forced_resolver(*a, **kw):
        return dict(forced) if forced else real_resolver(*a, **kw)

    fe_mod.get_triton_config = _forced_resolver

    torch.manual_seed(0)
    dev = "cuda"
    w13 = torch.empty(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=dev
    ).normal_(std=0.02)
    w2 = torch.empty(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=dev
    ).normal_(std=0.02)
    router = torch.empty(
        NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=dev
    ).normal_()
    topk_softmax = TopKSoftmax()
    fused = FusedExperts()

    m_keys = [1, 32, 256, 2048] if args.quick else M_KEYS
    grid = _grid(args.quick)
    best: dict[int, dict] = {}

    print(f"  E={NUM_EXPERTS} top_k={TOP_K} H={HIDDEN} I={INTERMEDIATE} "
          f"bf16  {len(grid)} configs x {len(m_keys)} M")
    print(f"  {'M':>6} {'ROWS/EXPERT':>12} {'HEURISTIC us':>13} "
          f"{'BEST us':>9} {'GAIN':>7}  BEST CONFIG")

    for m in m_keys:
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        logits = x @ router.t()
        weights, ids = topk_softmax(logits, TOP_K, renormalize=False)
        weights = weights.to(x.dtype)

        def run(cfg):
            forced.clear()
            forced.update(cfg)
            return fused(x, w13, w2, weights, ids, NUM_EXPERTS)

        heuristic = _get_default_config(m, NUM_EXPERTS, INTERMEDIATE)
        base_us = _time(lambda: run(dict(heuristic)), args.iters)

        best_us, best_cfg = base_us, dict(heuristic)
        for cfg in grid:
            try:
                us = _time(lambda: run(dict(cfg)), args.iters)
            except Exception:
                continue
            if us < best_us:
                best_us, best_cfg = us, dict(cfg)
        best[m] = best_cfg
        print(f"  {m:>6} {m * TOP_K / NUM_EXPERTS:>12.1f} {base_us:>13.1f} "
              f"{best_us:>9.1f} {base_us / best_us:>6.2f}x  {best_cfg}")

    if args.write:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "fastkernels", "tasks", "baseline", "L1", "moe_configs",
        )
        os.makedirs(out_dir, exist_ok=True)
        name = _get_config_file_name(NUM_EXPERTS, INTERMEDIATE, None)
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in sorted(best.items())}, f, indent=2)
            f.write("\n")
        print(f"\n  wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
