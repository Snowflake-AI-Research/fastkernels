"""CLI entry point for the fastkernels kernel benchmark suite.

Usage:
    # Test all operators that have candidates
    python -m fastkernels.bench.kernels

    # Test a specific operator
    python -m fastkernels.bench.kernels --target rms_norm

    # Filter to a specific model family
    python -m fastkernels.bench.kernels --target rms_norm --model llama

    # Filter to TP=4 scenarios only
    python -m fastkernels.bench.kernels --target rms_norm --tp 4

    # Restrict to LLM category
    python -m fastkernels.bench.kernels --target rms_norm --category llm

    # List available targets
    python -m fastkernels.bench.kernels --list
    python -m fastkernels.bench.kernels --list --level 1
"""

from __future__ import annotations

import argparse
import sys

from fastkernels.infra.kernel_swapper import (
    list_targets,
    print_model_operator_map,
)

from .result import KernelBenchResult
from .runner import run_all_kernel_benchmarks, run_kernel_benchmark

from fastkernels import run_output_path


def main():
    parser = argparse.ArgumentParser(
        description="fastkernels CUDA kernel benchmark suite (isolated forward() testing)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmark targets and exit",
    )
    parser.add_argument(
        "--map", action="store_true",
        help="Print operators-by-model and models-by-operator mappings",
    )
    parser.add_argument(
        "--level", type=int, default=None,
        help="Filter targets by level (1-4) when listing",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Benchmark target name (e.g. 'rms_norm', 'attention'). "
             "Omit to test all operators with candidates.",
    )
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="Filter scenarios by model key prefix (e.g. 'llama31' 'mixtral')",
    )
    parser.add_argument(
        "--tp", type=int, nargs="+", default=None,
        help="Filter scenarios by TP degree(s) (e.g. 1 4)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter scenarios by category (e.g. 'llm', 'vision')",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--num-runs", type=int, default=100,
        help="Number of timed runs for median (default: 100)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save JSON results (default: bench/results/kernels_<timestamp>.json)",
    )
    parser.add_argument(
        "--pytorch-reference", action="store_true", default=False,
        help="Compare production baseline against tasks/reference/L*/ PyTorch reference",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["candidate", "baseline_identity", "pytorch_reference", "candidate_smoke"],
        default="candidate",
        help=(
            "What to compare against the production baseline. "
            "baseline_identity validates the harness, pytorch_reference validates "
            "semantic references, and candidate_smoke does a one-run candidate check."
        ),
    )
    args = parser.parse_args()

    if args.map:
        print_model_operator_map()
        return

    if args.list:
        targets = list_targets(level=args.level)
        if not targets:
            print("No targets found.")
            return
        print(f"\n{'Level':<7} {'Name':<25} {'Class':<25} Models")
        print("-" * 80)
        for t in targets:
            print(
                f"  L{t.level:<5} {t.name:<25} {t.target_cls.__name__:<25} "
                f"{','.join(t.models)}"
            )
        print(f"\n{len(targets)} targets total.")
        return

    output_path = args.output_json or str(run_output_path("kernels"))

    from fastkernels.bench.tracking import tracker

    run_name = f"kernels_{args.target or 'all'}"
    bench_params = {
        "target": args.target or "all",
        "model_filter": str(args.model) if args.model else None,
        "tp_filter": str(args.tp) if args.tp else None,
        "category": args.category,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "pytorch_reference": args.pytorch_reference,
        "validation_mode": args.validation_mode,
    }

    with tracker.start_run(run_name, params=bench_params, tags={"tier": "kernel"}):
        if args.target is not None:
            op_result = run_kernel_benchmark(
                target_name=args.target,
                models=args.model,
                tp=args.tp,
                category=args.category,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
                pytorch_reference=args.pytorch_reference,
                validation_mode=args.validation_mode,
            )

            result = KernelBenchResult(operators=[op_result])
            result.compute_aggregates()
        else:
            result = run_all_kernel_benchmarks(
                models=args.model,
                tp=args.tp,
                category=args.category,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
                pytorch_reference=args.pytorch_reference,
                validation_mode=args.validation_mode,
            )

        tracker.log_kernel_bench(result)

        result.print_table(single_target=(args.target is not None))
        result.save_json(output_path)
        print(f"\n  Results saved to: {output_path}")

    sys.exit(0 if result.all_passed() else 1)


if __name__ == "__main__":
    main()
