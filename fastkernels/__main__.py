"""CLI entry point for fastkernels."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="fastkernels CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ``fastkernels list`` command
    list_parser = subparsers.add_parser("list", help="List families, architectures, and benchmarks")

    args = parser.parse_args()

    if args.command == "list":
        from .list import main as list_main
        list_main()


if __name__ == "__main__":
    main()
