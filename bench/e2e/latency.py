"""Single-batch latency benchmark for kb-nano.

Modeled after ``vllm bench latency``. Runs a fixed batch of requests repeatedly
and measures per-iteration latency with percentile reporting.

Usage:
    python -m kb_nano.bench.e2e latency \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --batch-size 8 --input-len 512 --output-len 128 \\
        --num-iters 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from kb_nano.infra.kernel_swapper import (
    apply_candidates,
    discover_candidates,
    print_candidate_summary,
)


def validate_args(args: argparse.Namespace):
    """Validate CLI arguments for the latency benchmark."""
    if args.input_len <= 0:
        raise ValueError("--input-len must be > 0")
    if args.output_len <= 0:
        raise ValueError("--output-len must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_iters <= 0:
        raise ValueError("--num-iters must be > 0")
    if args.num_iters_warmup < 0:
        raise ValueError("--num-iters-warmup must be >= 0")
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.top_p is not None and not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1]")


def add_cli_args(parser: argparse.ArgumentParser):
    """Add latency-specific CLI arguments."""
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of sequences per batch (default: 8)",
    )
    parser.add_argument(
        "--input-len", type=int, default=512,
        help="Input prompt length in tokens (default: 512)",
    )
    parser.add_argument(
        "--output-len", type=int, default=128,
        help="Output length in tokens (default: 128)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0, matching vLLM. "
             "Use 0.0 for greedy/deterministic)",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", default=False,
        help="Disable CUDA graphs / torch.compile (default: False = full speed)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num-iters-warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--num-iters", type=int, default=30,
        help="Number of timed iterations (default: 30)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save latency results in JSON format",
    )
    parser.add_argument(
        "--save-outputs", type=str, default=None,
        help="Path to save generated outputs from the last iteration",
    )
    parser.add_argument(
        "--no-candidate-kernels", action="store_true", default=False,
        help="Disable candidate kernel auto-detection; use only baseline kernels",
    )


def main(args: argparse.Namespace):
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    undo_info = None
    if not args.no_candidate_kernels:
        candidates = discover_candidates()
        if candidates:
            print_candidate_summary(candidates)
            undo_info = apply_candidates(candidates)

    from kb_nano.infra.engine import LlamaEngine, SamplingParams

    print("=" * 70)
    print("  kb-nano Latency Benchmark")
    print("=" * 70)
    print(f"  Model          : {args.model}")
    print(f"  TP             : {args.tp}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Input length   : {args.input_len}")
    print(f"  Output length  : {args.output_len}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Top-p          : {args.top_p}")
    print(f"  Enforce eager  : {args.enforce_eager}")
    print(f"  Seed           : {args.seed}")
    print(f"  Warmup iters   : {args.num_iters_warmup}")
    print(f"  Timed iters    : {args.num_iters}")
    print("=" * 70)

    engine = LlamaEngine(
        model_name=args.model,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
    )

    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    ).tolist()

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.output_len,
        ignore_eos=True,
        seed=args.seed,
    )

    def run_once():
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(dummy_prompt_token_ids, sp)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return elapsed, outputs

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_once()

    print("Benchmarking...")
    latencies = []
    last_outputs = None
    for _ in tqdm(range(args.num_iters), desc="Bench iterations"):
        elapsed, outputs = run_once()
        latencies.append(elapsed)
        last_outputs = outputs

    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Avg latency: {np.mean(latencies):.4f} seconds")
    for pct, val in zip(percentages, percentiles):
        print(f"  P{pct:<3} latency: {val:.4f} seconds")
    print(f"{'=' * 70}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "tp": args.tp,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enforce_eager": args.enforce_eager,
        "num_iters_warmup": args.num_iters_warmup,
        "num_iters": args.num_iters,
        "avg_latency": float(np.mean(latencies)),
        "latencies": latencies.tolist(),
        "percentiles": {str(p): float(v) for p, v in zip(percentages, percentiles)},
    }

    # Log to MLflow
    from kb_nano.bench.tracking import tracker

    tracker.log_e2e(results, bench_type="latency")

    if args.output_json:
        output_json = args.output_json
    else:
        from kb_nano import run_output_path
        output_json = str(run_output_path("latency"))

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_json}")

    if args.save_outputs and last_outputs is not None:
        output_data = {
            **results,
            "outputs": [
                {
                    "generated_text": o.generated_text,
                    "token_ids": o.token_ids,
                }
                for o in last_outputs
            ],
        }
        os.makedirs(os.path.dirname(args.save_outputs) or ".", exist_ok=True)
        with open(args.save_outputs, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Outputs saved to: {args.save_outputs}")

    engine._cleanup()
    del engine

    if undo_info is not None:
        from kb_nano.infra.kernel_swapper import restore
        restore(undo_info)
