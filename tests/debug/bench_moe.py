#!/usr/bin/env python3
"""
MoE-layer microbenchmark: times just the MixtralMoE forward pass
at various batch sizes, isolating kernel performance from engine overhead.

Usage:
    python tests/bench_moe.py
    python tests/bench_moe.py --batch-sizes 1 4 16 64 256 --warmup 10 --iters 100
    python tests/bench_moe.py --tp 4
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

WORKER = r'''
import json, os, sys, time
with open(sys.argv[1]) as f:
    cfg = json.load(f)
sys.path.insert(0, cfg["project_root"])

def main():
    import torch
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    if cfg.get("pytorch_reference", False):
        from fastkernels.infra.kernel_swapper import (
            apply_candidates,
            discover_references,
            print_reference_summary,
        )
        references = discover_references()
        if references:
            print_reference_summary(references)
            apply_candidates(references)

    from fastkernels.tasks.baseline.L2.mixtral_moe import MixtralMoE
    from fastkernels.tasks.baseline.L4.mixtral import MixtralConfig

    tp = cfg.get("tp", 1)

    class FakeConfig:
        num_local_experts = 8
        num_experts_per_tok = 2
        hidden_size = 4096
        intermediate_size = 14336

    config = FakeConfig()

    moe = MixtralMoE(config).cuda().bfloat16()
    # Initialize with random weights
    moe.gate.weight.data.normal_()
    moe.w13.data.normal_(std=0.02)
    moe.w2.data.normal_(std=0.02)

    batch_sizes = cfg["batch_sizes"]
    warmup = cfg["warmup"]
    iters = cfg["iters"]

    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = moe(x)
        torch.cuda.synchronize()

        # Timed iterations
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            start_events[i].record()
            with torch.no_grad():
                _ = moe(x)
            end_events[i].record()

        torch.cuda.synchronize()

        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times_ms.sort()
        # Use median to avoid outliers
        median_ms = times_ms[len(times_ms) // 2]
        mean_ms = sum(times_ms) / len(times_ms)
        p10_ms = times_ms[len(times_ms) // 10]
        p90_ms = times_ms[int(len(times_ms) * 0.9)]

        results[str(bs)] = {
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "p10_ms": p10_ms,
            "p90_ms": p90_ms,
        }

    with open(cfg["output_file"], "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''


def main():
    parser = argparse.ArgumentParser(description="MoE layer microbenchmark")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Patch semantic PyTorch references from tasks/reference/L*/ into fastkernels.",
    )
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(package_dir)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp",
    ) as f:
        f.write(WORKER)
        script_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    config = {
        "project_root": project_root,
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "iters": args.iters,
        "tp": args.tp,
        "pytorch_reference": args.pytorch_reference,
        "output_file": output_path,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp",
    ) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        print("=" * 60)
        print("  MoE Layer Microbenchmark")
        print("=" * 60)
        print(f"  Batch sizes : {args.batch_sizes}")
        print(f"  Warmup      : {args.warmup}")
        print(f"  Iterations  : {args.iters}")
        print("=" * 60)

        result = subprocess.run(
            [sys.executable, script_path, config_path],
            timeout=600,
        )
        if result.returncode != 0:
            print("  ERROR: Worker failed")
            sys.exit(1)

        with open(output_path) as f:
            results = json.load(f)

        print(f"\n  {'BS':>6}  {'Median':>10}  {'Mean':>10}  {'P10':>10}  {'P90':>10}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
        for bs in args.batch_sizes:
            r = results[str(bs)]
            print(
                f"  {bs:>6}  {r['median_ms']:>9.3f}ms"
                f"  {r['mean_ms']:>9.3f}ms"
                f"  {r['p10_ms']:>9.3f}ms"
                f"  {r['p90_ms']:>9.3f}ms"
            )
        print("=" * 60)

    finally:
        os.unlink(script_path)
        os.unlink(config_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == "__main__":
    main()
