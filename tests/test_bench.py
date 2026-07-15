#!/usr/bin/env python3
"""
Test suite for the benchmarking infrastructure.

Sections 4-6: Unit tests (no GPU required).
Sections 8-9: Integration tests (GPU required and run by default).

Usage:
    python tests/test_bench.py                 # all tests
    python tests/test_bench.py --section 5      # run only section 5
"""

from __future__ import annotations

import argparse
import io
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
PACKAGE_NAME = os.path.basename(PACKAGE_DIR)

sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_pass_count = 0
_fail_count = 0


def check(condition: bool, label: str):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"    PASS  {label}")
    else:
        _fail_count += 1
        print(f"    FAIL  {label}")


class _Timeout:
    """POSIX alarm-based timeout guard for unit tests."""
    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *args):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("Unit test timed out")


# ===========================================================================
# Section 4: Standardized workloads (unit, no GPU)
# ===========================================================================
def test_section_4():
    print(f"\n{'=' * 60}")
    print("  SECTION 4: Standardized workloads")
    print(f"{'=' * 60}")

    from fastkernels.workloads import (
        DEFAULT_WORKLOAD_DATASETS,
        LATENCY_WORKLOADS,
        THROUGHPUT_WORKLOADS,
        get_max_seq_len,
    )

    # 4a. Throughput workload constants
    with _Timeout(30):
        check(len(THROUGHPUT_WORKLOADS) == 2, "4a. exactly 2 throughput workloads")
        names = [w.name for w in THROUGHPUT_WORKLOADS]
        check(
            names == ["mixed", "long-context"],
            f"4a. correct names: {names}",
        )
        mixed = THROUGHPUT_WORKLOADS[0]
        check(
            mixed.dataset_name.endswith("wildchat-mixed-1k") and mixed.decode_cap == 1024,
            "4a. mixed dataset configured (decode cap 1024)",
        )
        longctx = THROUGHPUT_WORKLOADS[1]
        check(
            longctx.dataset_name.endswith("longbench-longctx")
            and longctx.num_requests == 64 and longctx.decode_cap is None,
            "4a. long-context dataset configured (N=64, per-row output_len)",
        )

    # 4b. Latency workload constants
    with _Timeout(30):
        check(len(LATENCY_WORKLOADS) == 2, "4b. exactly 2 latency workloads")
        mixed_repo = DEFAULT_WORKLOAD_DATASETS["mixed"]
        sr = LATENCY_WORKLOADS[0]
        check(
            sr.name == "single-request" and sr.batch_size == 1
            and sr.output_len == 128 and sr.dataset_name == mixed_repo,
            "4b. single-request: bs=1, real mixed prompts, decode<=128",
        )
        fb = LATENCY_WORKLOADS[1]
        check(
            fb.name == "fixed-batch-32" and fb.batch_size == 32
            and fb.output_len == 128 and fb.dataset_name == mixed_repo,
            "4b. fixed-batch-32: bs=32, real mixed prompts, decode<=128",
        )

    # 4c. Immutability (frozen dataclasses)
    with _Timeout(30):
        try:
            THROUGHPUT_WORKLOADS[0].dataset_name = "other"
            check(False, "4c. throughput workloads should be immutable")
        except AttributeError:
            check(True, "4c. throughput workloads are frozen (immutable)")
        try:
            LATENCY_WORKLOADS[0].batch_size = 999
            check(False, "4c. latency workloads should be immutable")
        except AttributeError:
            check(True, "4c. latency workloads are frozen (immutable)")

    # 4d. get_max_seq_len
    with _Timeout(30):
        max_len = get_max_seq_len()
        check(
            max_len == 128,
            f"4d. latency decode-budget floor = {max_len} (expected 128)",
        )


# ===========================================================================
# Section 5: Multi-level conflict resolution (unit, no GPU)
# ===========================================================================
def _can_import_baseline() -> bool:
    """Whether the baseline operator modules import (needs torch + CUDA deps)."""
    try:
        importlib.import_module(f"{PACKAGE_NAME}.tasks.baseline.L1.rms_norm")
        return True
    except Exception as exc:
        print(f"    SKIP  baseline modules unavailable: {exc}")
        return False


def test_section_5():
    print(f"\n{'=' * 60}")
    print("  SECTION 5: Candidate operator discovery + swapping")
    print(f"{'=' * 60}")

    from fastkernels.list import (
        apply_candidates,
        discover_candidate_impls,
        discover_operator_targets,
        restore_candidates,
    )

    # 5a. Static operator discovery (pure ast, no torch needed).
    with _Timeout(30):
        targets = {t.name: t for t in discover_operator_targets()}
        check(len(targets) > 0, f"5a. discover_operator_targets() -> {len(targets)} ops")
        rms = targets.get("rms_norm")
        check(rms is not None and rms.level == 1, "5a. rms_norm discovered at L1")
        check(rms is not None and "llama31" in rms.models,
              '5a. rms_norm attributed to model "llama31"')

    # 5b. Candidate discovery: with none present, returns an empty list cleanly.
    with _Timeout(30):
        pairs = discover_candidate_impls()
        check(isinstance(pairs, list),
              f"5b. discover_candidate_impls() -> {len(pairs)} candidate(s)")

    # 5c. Swap + restore round-trip, including propagation into the higher-level
    #     baseline module that imports the swapped class (L1 -> L3).
    with _Timeout(60):
        if not _can_import_baseline():
            print("    SKIP  5c. baseline modules unavailable")
        else:
            name = targets["rms_norm"].class_name
            rms_mod = importlib.import_module(f"{PACKAGE_NAME}.tasks.baseline.L1.rms_norm")
            rms_cls = getattr(rms_mod, name)
            # The L3 decoder imports RMSNorm; loading it binds the reference the
            # swap must reach.
            decoder_mod = importlib.import_module(
                f"{PACKAGE_NAME}.tasks.baseline.L3.llama_decoder")

            class FakeRMSNorm(nn.Module):
                pass

            undo = apply_candidates([(targets["rms_norm"], rms_cls, FakeRMSNorm)])
            check(getattr(rms_mod, name) is FakeRMSNorm,
                  "5c. candidate swap applied at L1")
            check(getattr(decoder_mod, name, None) is FakeRMSNorm,
                  "5c. L3 baseline picks up the L1 swap")
            restore_candidates(undo)
            check(getattr(rms_mod, name) is rms_cls, "5c. restore works")


# ===========================================================================
# Section 6: CLI argument parsing (unit, no GPU)
# ===========================================================================
def test_section_6():
    print(f"\n{'=' * 60}")
    print("  SECTION 6: CLI argument parsing")
    print(f"{'=' * 60}")

    # 6b. eval CLI (fastkernels.eval)
    with _Timeout(30):
        result = subprocess.run(
            [sys.executable, "-m", "fastkernels.eval", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        if result.returncode != 0 and ("sgl_kernel" in result.stderr or "ModuleNotFoundError" in result.stderr):
            print("    SKIP  6c. sgl_kernel not available for subprocess import")
        else:
            check(
                result.returncode == 0 and "scenarios" in result.stdout
                and "--self-test" in result.stdout,
                "6c. eval CLI takes a scenarios table and --self-test",
            )
            check(
                "--max-requests" in result.stdout and "--max-layers" in result.stdout
                and "--gpus" in result.stdout,
                "6c. eval CLI accepts --max-requests, --max-layers, --gpus",
            )

    # 6d. Default JSON output path
    with _Timeout(30):
        from fastkernels import RESULTS_DIR, run_output_path

        kernels_default = run_output_path("kernels")
        check(
            kernels_default.parent == RESULTS_DIR
            and kernels_default.name.startswith("kernels_")
            and kernels_default.suffix == ".json",
            f"6d. kernels default output: {kernels_default}",
        )

        eval_default = run_output_path("eval")
        check(
            eval_default.parent == RESULTS_DIR
            and eval_default.name.startswith("eval_")
            and eval_default.suffix == ".json",
            f"6d. eval default output: {eval_default}",
        )


# ===========================================================================
# Section 9: Eval integration (GPU required, single GPU)
# ===========================================================================
def test_section_9():
    print(f"\n{'=' * 60}")
    print("  SECTION 9: Eval integration (GPU required)")
    print(f"{'=' * 60}")

    # 9a. Self-test: eval a scenario with baseline as the candidate. The two
    #     runs are identical, so correctness must be a perfect match. Capped to
    #     the first few layers / a handful of requests to keep the run short.
    with _Timeout(660):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "eval.json")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "fastkernels.eval", "minimal",
                     "--self-test",
                     "--max-requests", "8",
                     "--max-layers", "4",
                     "--gpus", "0",
                     "--output", json_path],
                    timeout=600, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                    env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
                )
                if result.returncode != 0:
                    print(f"    STDERR: {result.stderr[-500:]}")
                    check(False, "9a. eval --self-test subprocess failed")
                else:
                    check(os.path.exists(json_path), "9a. eval JSON created")
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data.get("self_test") is True
                            and "throughput" in data
                            and "correctness_passed" in data,
                            "9a. eval JSON has the expected schema",
                        )
                        check(
                            data.get("correctness_passed") is True,
                            "9a. self-test correctness passes (baseline == baseline)",
                        )
            except subprocess.TimeoutExpired:
                check(False, "9a. eval timed out after 600s")

    # 9b. --max-layers is validated (fast, no GPU work).
    with _Timeout(30):
        result = subprocess.run(
            [sys.executable, "-m", "fastkernels.eval", "minimal", "--max-layers", "0"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        check(
            result.returncode != 0 and "--max-layers must be >= 1" in result.stderr,
            "9b. eval rejects --max-layers < 1",
        )


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test the fastkernels benchmarking infrastructure",
    )
    parser.add_argument(
        "--section", type=int, default=None,
        help="Run only a specific section (4, 5, 6, 9)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  fastkernels benchmarking infrastructure tests")
    print("=" * 60)

    sections = {
        4: ("Standardized workloads", test_section_4),
        5: ("Conflict resolution", test_section_5),
        6: ("CLI argument parsing", test_section_6),
        9: ("Eval integration", test_section_9),
    }

    for num, (name, func) in sorted(sections.items()):
        if args.section is not None and args.section != num:
            continue
        try:
            func()
        except TimeoutError:
            check(False, f"Section {num} ({name}) timed out entirely")
        except Exception as e:
            check(False, f"Section {num} ({name}) raised exception: {e}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_pass_count} passed, {_fail_count} failed")
    print(f"{'=' * 60}")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
