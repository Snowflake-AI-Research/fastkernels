"""CLI entry point for the kb-nano eval sweep (Tier 3).

Usage:
    # Full eval (all models with candidate kernels, all categories)
    python -m kb_nano.bench.eval

    # Single category
    python -m kb_nano.bench.eval --category llm

    # Specific model
    python -m kb_nano.bench.eval --model meta-llama/Llama-3.1-8B-Instruct

    # Restrict TP degrees
    python -m kb_nano.bench.eval --tp 1 4

    # Custom output path (default: bench/results/eval_<timestamp>.json)
    python -m kb_nano.bench.eval --output-json results/my_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import EvalConfig
from .runner import run_eval

from kb_nano import run_output_path


def main():
    parser = argparse.ArgumentParser(
        prog="python -m kb_nano.bench.eval",
        description="kb-nano evaluation sweep: standardized multi-model benchmarking",
    )
    parser.add_argument(
        "--model", type=str, nargs="*", default=None,
        help="HuggingFace model name(s). Default: auto-detect from candidates.",
    )
    parser.add_argument(
        "--tp", type=int, nargs="+", default=[1, 4],
        help="TP degree(s) to evaluate (default: 1 4)",
    )
    parser.add_argument(
        "--category", type=str, nargs="*", default=None,
        help="Filter by category (e.g. 'llm', 'vision'). Default: all.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000,
        help="Number of prompts per throughput workload (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic alignment)",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", default=False,
        help="Disable CUDA graphs",
    )
    parser.add_argument(
        "--gpu-pool", type=int, default=8,
        help="Number of GPUs available for scheduling (default: 8)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save JSON results (default: bench/results/eval_<timestamp>.json)",
    )
    args = parser.parse_args()

    _default_output = str(run_output_path("eval"))

    config = EvalConfig(
        models=args.model,
        tp_degrees=args.tp,
        categories=args.category,
        seed=args.seed,
        temperature=args.temperature,
        enforce_eager=args.enforce_eager,
        output_json=args.output_json or _default_output,
        num_prompts=args.num_prompts,
    )

    from kb_nano.bench.tracking import tracker

    eval_params = {
        "models": str(args.model) if args.model else "auto",
        "tp_degrees": str(args.tp),
        "categories": str(args.category) if args.category else "all",
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "temperature": args.temperature,
    }

    with tracker.start_run("eval", params=eval_params, tags={"tier": "eval"}):
        report = asyncio.run(run_eval(config, gpu_pool=args.gpu_pool))
        tracker.log_eval(report)

        report.print_table()

        output_path = args.output_json or _default_output
        report.save_json(output_path)
        print(f"\n  Results saved to: {output_path}")

    sys.exit(1 if report.failed_jobs > 0 else 0)


if __name__ == "__main__":
    main()
