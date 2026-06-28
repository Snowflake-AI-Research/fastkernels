"""Data classes for kernel benchmark results."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class ScenarioResult:
    """Result from a single scenario of a kernel benchmark."""
    name: str
    correct: bool
    max_error_ratio: float
    mean_abs_diff: float
    baseline_ms: float
    candidate_ms: float
    speedup: float
    failure_reason: str | None = None
    classification: str | None = None


@dataclass
class OperatorResult:
    """Aggregated result for a single operator across all scenarios."""
    target: str
    level: int
    candidate_path: str
    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    avg_max_error_ratio: float = 0.0
    avg_mean_abs_diff: float = 0.0
    avg_speedup: float = 0.0
    scenarios: list[ScenarioResult] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        self.total_scenarios = len(self.scenarios)
        self.passed = sum(1 for s in self.scenarios if s.correct)
        self.failed = self.total_scenarios - self.passed
        if self.scenarios:
            self.avg_max_error_ratio = sum(s.max_error_ratio for s in self.scenarios) / len(self.scenarios)
            self.avg_mean_abs_diff = sum(s.mean_abs_diff for s in self.scenarios) / len(self.scenarios)
            self.avg_speedup = sum(s.speedup for s in self.scenarios) / len(self.scenarios)


@dataclass
class KernelBenchResult:
    """Top-level result containing all operators tested."""
    total_operators: int = 0
    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    avg_max_error_ratio: float = 0.0
    avg_mean_abs_diff: float = 0.0
    avg_speedup: float = 0.0
    operators: list[OperatorResult] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        for op in self.operators:
            op.compute_aggregates()
        self.total_operators = len(self.operators)
        self.total_scenarios = sum(op.total_scenarios for op in self.operators)
        self.passed = sum(op.passed for op in self.operators)
        self.failed = sum(op.failed for op in self.operators)
        if self.total_scenarios > 0:
            all_ratios = [s.max_error_ratio for op in self.operators for s in op.scenarios]
            all_diffs = [s.mean_abs_diff for op in self.operators for s in op.scenarios]
            all_speedups = [s.speedup for op in self.operators for s in op.scenarios]
            self.avg_max_error_ratio = sum(all_ratios) / len(all_ratios)
            self.avg_mean_abs_diff = sum(all_diffs) / len(all_diffs)
            self.avg_speedup = sum(all_speedups) / len(all_speedups)

    def all_passed(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_table(self, single_target: bool = False) -> None:
        """Print human-readable terminal output."""
        for op in self.operators:
            impl_label = (
                "PyTorch reference"
                if op.candidate_path.startswith("tasks/reference/")
                else "Candidate"
            )
            print(f"\n{'=' * 70}")
            print(f"  Kernel Benchmark: {op.target}")
            print(f"  {impl_label}: {op.candidate_path}")
            print(f"  Scenarios: {op.total_scenarios} (from InputRegistry)")
            print(f"{'=' * 70}")
            print()
            print(f"  {'SCENARIO':<40} {'CORRECT':>8}   {'ERR_RATIO':>10}   {'SPEEDUP':>8}")
            print(f"  {'─' * 66}")
            for s in op.scenarios:
                status = "PASS" if s.correct else "FAIL"
                print(
                    f"  {s.name:<40} {status:>8}   {s.max_error_ratio:>10.2e}   {s.speedup:>7.2f}x"
                )
            print(f"  {'─' * 66}")
            print(f"  OVERALL: {op.passed}/{op.total_scenarios} PASS    "
                  f"avg speedup: {op.avg_speedup:.2f}x")
            print(f"{'=' * 70}")

        if not single_target and len(self.operators) > 1:
            print(f"\n{'=' * 70}")
            print("  ALL OPERATORS SUMMARY")
            print(f"{'=' * 70}")
            print(
                f"  {'OPERATOR':<20} {'LEVEL':>5}   {'PASS/TOTAL':>10}   "
                f"{'AVG ERR_RATIO':>13}   {'AVG SPEEDUP':>11}   {'STATUS':>6}"
            )
            print(f"  {'─' * 68}")
            for op in self.operators:
                status = "PASS" if op.failed == 0 else "FAIL"
                print(
                    f"  {op.target:<20} L{op.level:<4}   "
                    f"{op.passed}/{op.total_scenarios:>7}   "
                    f"{op.avg_max_error_ratio:>13.2e}   "
                    f"{op.avg_speedup:>10.2f}x   "
                    f"{status:>6}"
                )
            print(f"  {'─' * 68}")
            print(
                f"  TOTAL: {self.passed}/{self.total_scenarios} PASS   "
                f"overall avg err_ratio: {self.avg_max_error_ratio:.2e}   "
                f"overall avg speedup: {self.avg_speedup:.2f}x"
            )
            print(f"{'=' * 70}")
