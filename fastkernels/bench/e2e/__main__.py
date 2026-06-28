"""CLI dispatcher for fastkernels e2e benchmarks.

Usage:
    python -m fastkernels.bench.e2e throughput [args...]
    python -m fastkernels.bench.e2e latency [args...]
    python -m fastkernels.bench.e2e serve [args...]
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="python -m fastkernels.bench.e2e",
        description="fastkernels end-to-end benchmarks",
    )
    subparsers = parser.add_subparsers(dest="command", help="Benchmark type")

    from fastkernels.bench.e2e import latency, serve, throughput

    tp = subparsers.add_parser("throughput", help="Offline throughput benchmark")
    throughput.add_cli_args(tp)

    lp = subparsers.add_parser("latency", help="Single-batch latency benchmark")
    latency.add_cli_args(lp)

    sp = subparsers.add_parser("serve", help="Online serving benchmark")
    serve.add_cli_args(sp)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    from fastkernels.bench.tracking import tracker

    model = getattr(args, "model", "unknown")
    tp = getattr(args, "tp", 1)
    e2e_params = {"model": model, "tp": tp, "bench_type": args.command}
    run_name = f"e2e_{args.command}_{model.split('/')[-1]}"

    with tracker.start_run(run_name, params=e2e_params, tags={"tier": "e2e"}):
        if args.command == "throughput":
            throughput.main(args)
        elif args.command == "latency":
            latency.main(args)
        elif args.command == "serve":
            serve.main(args)


if __name__ == "__main__":
    main()
