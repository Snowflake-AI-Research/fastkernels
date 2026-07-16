"""CLI entry point for fastkernels."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        from .capture import main as capture_main
        raise SystemExit(capture_main(sys.argv[2:]))

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        from .eval import main as eval_main
        raise SystemExit(eval_main(sys.argv[2:]))

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        from .list import main as list_main
        list_main(sys.argv[2:])
        raise SystemExit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        from .bench_kernel import main as bench_main
        raise SystemExit(bench_main(sys.argv[2:]))

    if len(sys.argv) > 1 and sys.argv[1] == "create-stubs":
        from .utils.create_stubs import main as create_stubs_main
        raise SystemExit(create_stubs_main(sys.argv[2:]))

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        from .validate._cli import main as validate_main
        raise SystemExit(validate_main(sys.argv[2:]))

    parser = argparse.ArgumentParser(description="fastkernels CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ``fastkernels list`` command (args forwarded to ``fastkernels.list``)
    subparsers.add_parser(
        "list",
        help="List families/architectures/benchmarks, or the model<->operator "
             "map with '--map'",
    )

    subparsers.add_parser(
        "capture",
        help="Capture runtime init/forward type, shape, and dtype metadata",
    )

    subparsers.add_parser(
        "eval",
        help="Evaluate candidate task implementations against the baseline "
             "end-to-end for correctness and performance across a scenarios "
             "table (see 'fastkernels eval --help')",
    )

    subparsers.add_parser(
        "bench",
        help="Benchmark candidate kernels against their baseline for "
             "correctness and performance (see 'fastkernels bench --help')",
    )

    subparsers.add_parser(
        "create-stubs",
        help="Scaffold candidate-kernel skeleton modules under tasks/candidate/ "
             "(see 'fastkernels create-stubs --help')",
    )

    subparsers.add_parser(
        "validate",
        help="Validate models in a scenario table against their SOTA reference "
             "library by running the proper bench harness per model "
             "(see 'fastkernels validate --help')",
    )

    parser.parse_args()


if __name__ == "__main__":
    main()
