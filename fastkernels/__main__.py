"""CLI entry point for fastkernels."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        from .capture import main as capture_main
        raise SystemExit(capture_main(sys.argv[2:]))

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        from .list import main as list_main
        list_main(sys.argv[2:])
        raise SystemExit(0)

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

    parser.parse_args()


if __name__ == "__main__":
    main()
