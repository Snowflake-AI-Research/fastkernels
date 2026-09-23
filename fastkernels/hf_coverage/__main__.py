"""Run matching HF and FastKernels workloads in separate environments."""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="modeling-file identifier in corpus.json")
    parser.add_argument("--list", action="store_true", help="list implemented cases")
    parser.add_argument("--variant", help="declared workload variant, required for entries with multiple workloads")
    parser.add_argument("--hf-python", default=os.environ.get("HF_COVERAGE_REFERENCE_PYTHON"))
    parser.add_argument("--hf-source", default=os.environ.get("HF_COVERAGE_REFERENCE_SOURCE"),
                        help="pinned Transformers checkout or source directory, used only by reference workers")
    parser.add_argument("--output-dir", help="new or empty directory under your scratch storage")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                        help="execution dtype; defaults to the case's declared dtype, otherwise bfloat16")
    parser.add_argument("--ref-attn", choices=("eager", "sdpa"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--worker-timeout", type=int, default=1800)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--upcast-from", help="prior low-precision run directory; reuse its rounded values in FP32")
    reuse.add_argument("--reuse-from", help="prior run with the same case, dtype, and seed; share its prepared file")
    parser.add_argument("--state-dict", help="optional file of common weights in the pinned HF model's state_dict format")
    parser.add_argument("--input-dict", help="optional file of processor-prepared common input tensors")
    args = parser.parse_args()
    from .cases import CASES

    if args.list:
        print("\n".join(sorted(CASES)))
        return 0
    if not args.model or args.model not in CASES:
        parser.error("select an implemented model; use --list to see available cases")
    from .runner import run_case, select_case

    try:
        case = select_case(args.model, args.variant)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dtype is None:
        args.dtype = case.get("default_dtype", "bfloat16")
    if not args.hf_python or not args.output_dir:
        parser.error("--hf-python and --output-dir are required for a run")
    if args.warmup < 0 or args.iterations < 1 or args.worker_timeout < 1:
        parser.error("warmup must be nonnegative; iterations and worker-timeout must be positive")
    if args.upcast_from and args.dtype != "float32":
        parser.error("--upcast-from requires --dtype float32")
    if (args.upcast_from or args.reuse_from) and (args.state_dict or args.input_dict):
        parser.error("a prior run already supplies weights and inputs; additional files cannot be combined with it")
    return run_case(args)


if __name__ == "__main__":
    raise SystemExit(main())
