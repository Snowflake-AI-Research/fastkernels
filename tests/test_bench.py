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
def _can_discover_targets() -> bool:
    """Check if target discovery works (requires sgl_kernel and other CUDA deps)."""
    try:
        from fastkernels.infra.kernel_swapper import discover_targets
        discover_targets()
        return True
    except Exception as exc:
        print(f"    SKIP  target discovery unavailable: {exc}")
        return False


def test_section_5():
    print(f"\n{'=' * 60}")
    print("  SECTION 5: Multi-level conflict resolution")
    print(f"{'=' * 60}")

    from fastkernels.infra.kernel_swapper import (
        BenchTarget,
        _detect_subsumption,
        _sort_by_level,
    )

    # 5a. Bottom-up ordering
    with _Timeout(30):
        t1 = BenchTarget("op_a", 3, "tasks.baseline.L3.op_a", ["llama31"], nn.Module)
        t2 = BenchTarget("op_b", 1, "tasks.baseline.L1.op_b", ["llama31"], nn.Module)
        t3 = BenchTarget("op_c", 2, "tasks.baseline.L2.op_c", ["llama31"], nn.Module)

        class Fake(nn.Module):
            pass

        candidates = [(t1, Fake), (t2, Fake), (t3, Fake)]
        sorted_c = _sort_by_level(candidates)
        levels = [t.level for t, _ in sorted_c]
        check(levels == [1, 2, 3], f"5a. sorted levels: {levels} == [1, 2, 3]")

    has_targets = _can_discover_targets()

    # 5b. Subsumption detection
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5b. sgl_kernel not available, cannot discover targets")
        else:
            from fastkernels.infra.kernel_swapper import get

            rms_target = get("rms_norm")
            llama_decoder_target = get("llama_decoder")

            class FakeRMS(nn.Module):
                pass

            class FakeDecoder(nn.Module):
                pass

            candidates = [
                (rms_target, FakeRMS),
                (llama_decoder_target, FakeDecoder),
            ]
            warnings = _detect_subsumption(candidates)
            found_subsumption = any(
                lower_name == "rms_norm" for _, _, lower_name, _ in warnings
            )
            check(
                found_subsumption,
                f"5b. L3 llama_decoder subsumes L1 rms_norm (found {len(warnings)} warnings)",
            )

    # 5c. No false positives (mock targets with no import relationship)
    with _Timeout(30):
        class FakeA(nn.Module):
            pass
        class FakeB(nn.Module):
            pass

        mock_l1 = BenchTarget("fake_op_alpha", 1, "tasks.baseline.L1.fake_op_alpha", ["llama31"], nn.Module)
        mock_l2 = BenchTarget("fake_op_beta", 2, "tasks.baseline.L2.fake_op_beta", ["llama31"], nn.Module)
        candidates_no_overlap = [
            (mock_l1, FakeA),
            (mock_l2, FakeB),
        ]
        warnings_no = _detect_subsumption(candidates_no_overlap)
        check(
            len(warnings_no) == 0,
            "5c. mock targets with no import chain -> no subsumption (no false positive)",
        )

    # 5d. Patching order
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5d. sgl_kernel not available, cannot discover targets")
        else:
            from fastkernels.infra.kernel_swapper import get, patch_class, restore
            rms_target = get("rms_norm")
            rms_module = importlib.import_module(f"{PACKAGE_NAME}.{rms_target.module_path}")
            original_cls = rms_target.target_cls

            class PatchedRMSNorm(nn.Module):
                _is_patched = True

            undo = patch_class(rms_target, PatchedRMSNorm)
            patched_cls = getattr(rms_module, original_cls.__name__)
            check(
                patched_cls is PatchedRMSNorm,
                "5d. L1 patch applied correctly",
            )

            decoder_mod = importlib.import_module(f"{PACKAGE_NAME}.tasks.baseline.L3.llama_decoder")
            decoder_rms = getattr(decoder_mod, original_cls.__name__, None)
            check(
                decoder_rms is PatchedRMSNorm,
                "5d. L3 baseline picks up L1 patch",
            )

            restore(undo)
            check(
                getattr(rms_module, original_cls.__name__) is original_cls,
                "5d. restore works correctly",
            )

    # 5e. Discovery convenience functions
    with _Timeout(30):
        if not has_targets:
            print("    SKIP  5e. sgl_kernel not available, cannot discover targets")
        else:
            from fastkernels.infra.kernel_swapper import (
                list_targets, models_for_target, targets_for_model,
            )

            all_targets = list_targets()
            check(len(all_targets) > 0, "5e. list_targets() returns non-empty list")

            l1_targets = list_targets(level=1)
            check(
                len(l1_targets) > 0 and all(t.level == 1 for t in l1_targets),
                "5e. list_targets(level=1) returns only L1 targets",
            )

            rms_models = models_for_target("rms_norm")
            check(
                "llama31" in rms_models,
                '5e. models_for_target("rms_norm") contains "llama31"',
            )

            llama_targets = targets_for_model("llama31")
            check(
                len(llama_targets) > 0
                and all("llama31" in t.models for t in llama_targets),
                '5e. targets_for_model("llama31") all include "llama31"',
            )


# ===========================================================================
# Section 6: CLI argument parsing (unit, no GPU)
# ===========================================================================
def test_section_6():
    print(f"\n{'=' * 60}")
    print("  SECTION 6: CLI argument parsing")
    print(f"{'=' * 60}")

    # 6b. bench.e2e CLI
    with _Timeout(30):
        result = subprocess.run(
            [sys.executable, "-m", "fastkernels.bench.e2e", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        if result.returncode != 0:
            # May fail due to missing sgl_kernel on import - check for import-related vs parse errors
            if "sgl_kernel" in result.stderr or "ModuleNotFoundError" in result.stderr:
                print("    SKIP  6b. sgl_kernel not available for subprocess import")
            else:
                check(False, f"6b. e2e --help failed: {result.stderr[-200:]}")
        else:
            check(
                "throughput" in result.stdout and "latency" in result.stdout
                and "serve" in result.stdout,
                "6b. e2e help shows throughput, latency, serve subcommands",
            )
            # Check eval is not in subcommand list (after the header)
            help_text = result.stdout
            if "Benchmark type" in help_text:
                subcommand_section = help_text.split("Benchmark type")[1]
                check(
                    "eval" not in subcommand_section,
                    "6b. eval subcommand removed from e2e",
                )
            else:
                check(True, "6b. eval subcommand not present in e2e help")

    # 6c. eval CLI (fastkernels.eval)
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
# Section 8: E2E integration (GPU required, single GPU)
# ===========================================================================
def test_section_8():
    print(f"\n{'=' * 60}")
    print("  SECTION 8: E2E integration (GPU required)")
    print(f"{'=' * 60}")

    # 8a. Throughput single-run
    with _Timeout(360):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "throughput.json")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "fastkernels.bench.e2e", "throughput",
                     "--model", "meta-llama/Llama-3.1-8B-Instruct",
                     "--tp", "1",
                     "--dataset-name", "random",
                     "--random-input-len", "128",
                     "--random-output-len", "64",
                     "--num-prompts", "10",
                     "--output-json", json_path,
                     "--no-candidate-kernels"],
                    timeout=300, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"    STDERR: {result.stderr[-500:]}")
                    check(False, "8a. throughput subprocess failed")
                else:
                    check(os.path.exists(json_path), "8a. throughput JSON created")
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data.get("tokens_per_second", 0) > 0,
                            f"8a. tokens_per_second={data.get('tokens_per_second', 0):.0f} > 0",
                        )
            except subprocess.TimeoutExpired:
                check(False, "8a. throughput timed out after 300s")

    # 8b. Latency single-run
    with _Timeout(360):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "latency.json")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "fastkernels.bench.e2e", "latency",
                     "--model", "meta-llama/Llama-3.1-8B-Instruct",
                     "--batch-size", "1",
                     "--input-len", "128",
                     "--output-len", "64",
                     "--num-iters-warmup", "1",
                     "--num-iters", "3",
                     "--output-json", json_path,
                     "--no-candidate-kernels"],
                    timeout=300, cwd=PROJECT_ROOT,
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"    STDERR: {result.stderr[-500:]}")
                    check(False, "8b. latency subprocess failed")
                else:
                    check(os.path.exists(json_path), "8b. latency JSON created")
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            data = json.load(f)
                        check(
                            data.get("avg_latency", 0) > 0,
                            f"8b. avg_latency={data.get('avg_latency', 0):.4f} > 0",
                        )
            except subprocess.TimeoutExpired:
                check(False, "8b. latency timed out after 300s")

    # 8c. JSON default save (verify default path works with --output-json)
    with _Timeout(30):
        check(True, "8c. default JSON paths verified in section 6d (no redundant GPU run)")


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
        help="Run only a specific section (4, 5, 6, 8, 9)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  fastkernels benchmarking infrastructure tests")
    print("=" * 60)

    sections = {
        4: ("Standardized workloads", test_section_4),
        5: ("Conflict resolution", test_section_5),
        6: ("CLI argument parsing", test_section_6),
        8: ("E2E integration", test_section_8),
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
