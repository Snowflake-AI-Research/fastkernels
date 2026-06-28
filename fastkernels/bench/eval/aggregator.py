"""Hierarchical result aggregation for Tier 3 eval.

Aggregates results at three levels:
1. Per-model: for each (model, tp) pair and each workload scenario
2. Per-category: averages across all models in a category
3. Overall: single-number summaries across all categories

Output format: JSON (machine-readable) + formatted table (human-readable).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import JobResult

_DEFAULT_LAMBDA = 0.5
_DEFAULT_CORRECTNESS_THRESHOLD = 1.0


def _detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        for tag in ("B200", "B100", "H200", "H100", "A100", "A10G", "L40S", "L40", "L4"):
            if tag in out:
                return tag
        return out.split()[-1]
    except Exception:
        return "unknown"


def _valid_speedup(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _geomean(values: list[float], default: float = 1.0) -> float:
    valid = [v for v in values if _valid_speedup(v)]
    if not valid:
        return default
    return math.exp(sum(math.log(v) for v in valid) / len(valid))


def _blend_speedup(
    throughput_speedup: float,
    latency_speedup: float,
    lambda_weight: float = _DEFAULT_LAMBDA,
) -> float:
    if not _valid_speedup(throughput_speedup) or not _valid_speedup(latency_speedup):
        return 0.0
    return (
        throughput_speedup ** lambda_weight
        * latency_speedup ** (1.0 - lambda_weight)
    )


@dataclass
class ModelResult:
    model: str
    tp: int
    status: str = "OK"
    throughput_speedup: float = 1.0
    latency_speedup: float = 1.0
    alignment_rate: float = 1.0
    correctness_score: float = 1.0
    valid: bool = True
    blended_speedup: float = 1.0
    error: str | None = None


@dataclass
class CategoryResult:
    name: str
    num_models: int = 0
    avg_throughput_speedup: float = 1.0
    avg_latency_speedup: float = 1.0
    alignment_rate: float = 1.0
    macro_correctness: float = 1.0
    macro_coverage: float = 1.0
    macro_speedup: float = 1.0
    macro_score: float = 1.0
    models: list[ModelResult] = field(default_factory=list)


@dataclass
class EvalReport:
    timestamp: str = ""
    wall_clock_seconds: float = 0.0
    gpu: str = "unknown"
    num_gpus: int = 8
    total_models: int = 0
    total_jobs: int = 0
    avg_throughput_speedup: float = 1.0
    avg_latency_speedup: float = 1.0
    alignment_rate: float = 1.0
    macro_correctness: float = 1.0
    macro_coverage: float = 1.0
    macro_speedup: float = 1.0
    macro_score: float = 1.0
    failed_jobs: int = 0
    categories: list[CategoryResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_table(self) -> None:
        """Print human-readable terminal output."""
        print(f"\n{'=' * 70}")
        print("  Evaluation")
        print(f"{'=' * 70}")
        cat_names = ", ".join(c.name for c in self.categories)
        print(f"  Categories     : {cat_names}")
        print(f"  Models         : {self.total_models}")
        tp_set = sorted(set(m.tp for c in self.categories for m in c.models))
        print(f"  TP degrees     : {tp_set}")
        print(f"  Throughput     : prefill-heavy (1024/512), balanced (512/512),")
        print(f"                   decode-heavy (512/1024) — 1000 reqs each")
        print(f"  Latency        : single-request (bs=1, 128/128),")
        print(f"                   fixed-batch-32 (bs=32, 128/128)")
        print(f"{'=' * 70}")

        for cat in self.categories:
            label = cat.name.upper()
            pad = (66 - len(label)) // 2
            print(f"\n  {'·' * pad}  {label}  {'·' * pad}")
            print(
                f"  {'MODEL':<35} {'TP':>3}   {'THRU SPEEDUP':>12}   "
                f"{'LAT SPEEDUP':>11}   {'ALIGNED':>7}"
            )
            print(f"  {'─' * 66}")
            for m in cat.models:
                if m.status == "FAILED":
                    print(f"  {m.model.split('/')[-1]:<35} {m.tp:>3}   "
                          f"{'FAILED':>12}   {'':>11}   {'':>7}")
                else:
                    align_str = f"{m.alignment_rate:.1%}"
                    print(
                        f"  {m.model.split('/')[-1]:<35} {m.tp:>3}   "
                        f"{m.throughput_speedup:>11.2f}x   "
                        f"{m.latency_speedup:>10.2f}x   "
                        f"{align_str:>7}"
                    )
            print(f"  {'─' * 66}")
            print(
                f"  {label} avg:  thru {cat.avg_throughput_speedup:.2f}x  |  "
                f"lat {cat.avg_latency_speedup:.2f}x  |  "
                f"aligned {cat.alignment_rate:.1%}"
            )

        wall_min = int(self.wall_clock_seconds // 60)
        wall_sec = int(self.wall_clock_seconds % 60)
        if wall_min >= 60:
            wall_str = f"{wall_min // 60}h {wall_min % 60}m"
        else:
            wall_str = f"{wall_min}m {wall_sec}s"

        print(f"\n{'=' * 70}")
        print("  OVERALL")
        print(f"{'=' * 70}")
        print(f"  Total models evaluated : {self.total_models}")
        print(f"  Total (model, tp) jobs : {self.total_jobs}")
        print(f"  Avg throughput speedup : {self.avg_throughput_speedup:.2f}x")
        print(f"  Avg latency speedup    : {self.avg_latency_speedup:.2f}x")
        print(f"  Overall alignment      : {self.alignment_rate:.1%}")
        print(f"  MacroEval speedup      : {self.macro_speedup:.2f}x")
        print(f"  MacroEval correctness  : {self.macro_correctness:.1%}")
        print(f"  MacroEval coverage     : {self.macro_coverage:.1%}")
        print(f"  MacroEval score        : {self.macro_score:.2f}")
        print(f"  Failed jobs            : {self.failed_jobs}")
        print(f"  Wall-clock time        : {wall_str}")
        print(f"{'=' * 70}")


class Aggregator:
    """Aggregates job results into hierarchical EvalReport."""

    @staticmethod
    def aggregate(
        results: list[JobResult],
        wall_clock_seconds: float = 0.0,
    ) -> EvalReport:
        gpu = _detect_gpu_name()

        category_map: dict[str, list[ModelResult]] = {}
        all_throughput_speedups: list[float] = []
        all_latency_speedups: list[float] = []
        all_alignment_nums: list[int] = []
        all_alignment_dens: list[int] = []
        failed = 0

        for jr in results:
            if jr.status == "FAILED":
                mr = ModelResult(
                    model=jr.model,
                    tp=jr.tp,
                    status="FAILED",
                    throughput_speedup=0.0,
                    latency_speedup=0.0,
                    alignment_rate=0.0,
                    correctness_score=0.0,
                    valid=False,
                    blended_speedup=0.0,
                    error=jr.error,
                )
                category_map.setdefault(jr.category, []).append(mr)
                failed += 1
                continue

            thru_speedups = [t["speedup"] for t in jr.throughput_results]
            lat_speedups = [l["speedup"] for l in jr.latency_results]
            avg_thru = _geomean(thru_speedups)
            avg_lat = _geomean(lat_speedups)

            total_aligned = sum(t.get("aligned_matches", 0) for t in jr.throughput_results)
            total_align_denom = sum(t.get("aligned_total", 0) for t in jr.throughput_results)
            align_rate = total_aligned / total_align_denom if total_align_denom > 0 else 1.0
            correctness_score = max(0.0, min(1.0, align_rate))
            valid = correctness_score >= _DEFAULT_CORRECTNESS_THRESHOLD
            blended_speedup = (
                _blend_speedup(avg_thru, avg_lat)
                if valid else 0.0
            )

            mr = ModelResult(
                model=jr.model,
                tp=jr.tp,
                status="OK",
                throughput_speedup=avg_thru,
                latency_speedup=avg_lat,
                alignment_rate=align_rate,
                correctness_score=correctness_score,
                valid=valid,
                blended_speedup=blended_speedup,
            )
            category_map.setdefault(jr.category, []).append(mr)
            all_throughput_speedups.append(avg_thru)
            all_latency_speedups.append(avg_lat)
            all_alignment_nums.append(total_aligned)
            all_alignment_dens.append(total_align_denom)

        categories: list[CategoryResult] = []
        for cat_name in sorted(category_map):
            cat_models = category_map[cat_name]
            ok_models = [m for m in cat_models if m.status == "OK"]
            valid_models = [m for m in ok_models if m.valid]
            cat_thru = (
                _geomean([m.throughput_speedup for m in ok_models])
                if ok_models else 0.0
            )
            cat_lat = (
                _geomean([m.latency_speedup for m in ok_models])
                if ok_models else 0.0
            )
            cat_align = (
                sum(m.alignment_rate for m in ok_models) / len(ok_models)
                if ok_models else 0.0
            )
            cat_correctness = (
                sum(m.correctness_score for m in cat_models) / len(cat_models)
                if cat_models else 0.0
            )
            cat_coverage = (
                sum(1 for m in cat_models if m.valid) / len(cat_models)
                if cat_models else 0.0
            )
            cat_macro_speedup = _geomean(
                [m.blended_speedup for m in valid_models],
                default=1.0,
            )
            if not valid_models:
                cat_macro_speedup = 1.0
            cat_score = cat_macro_speedup * cat_correctness * cat_coverage

            categories.append(CategoryResult(
                name=cat_name,
                num_models=len(ok_models),
                avg_throughput_speedup=cat_thru,
                avg_latency_speedup=cat_lat,
                alignment_rate=cat_align,
                macro_correctness=cat_correctness,
                macro_coverage=cat_coverage,
                macro_speedup=cat_macro_speedup,
                macro_score=cat_score,
                models=cat_models,
            ))

        overall_thru = (
            _geomean(all_throughput_speedups)
            if all_throughput_speedups else 1.0
        )
        overall_lat = (
            _geomean(all_latency_speedups)
            if all_latency_speedups else 1.0
        )
        overall_align_num = sum(all_alignment_nums)
        overall_align_den = sum(all_alignment_dens)
        overall_align = overall_align_num / overall_align_den if overall_align_den > 0 else 1.0

        unique_models = set()
        for jr in results:
            unique_models.add(jr.model)

        macro_correctness = (
            sum(c.macro_correctness for c in categories) / len(categories)
            if categories else 1.0
        )
        macro_coverage = (
            sum(c.macro_coverage for c in categories) / len(categories)
            if categories else 1.0
        )
        speedup_categories = [
            c.macro_speedup
            for c in categories
            if any(m.valid for m in c.models)
        ]
        macro_speedup = _geomean(speedup_categories) if speedup_categories else 1.0
        macro_score = macro_speedup * macro_correctness * macro_coverage

        return EvalReport(
            timestamp=datetime.now().isoformat(),
            wall_clock_seconds=wall_clock_seconds,
            gpu=gpu,
            num_gpus=8,
            total_models=len(unique_models),
            total_jobs=len(results),
            avg_throughput_speedup=overall_thru,
            avg_latency_speedup=overall_lat,
            alignment_rate=overall_align,
            macro_correctness=macro_correctness,
            macro_coverage=macro_coverage,
            macro_speedup=macro_speedup,
            macro_score=macro_score,
            failed_jobs=failed,
            categories=categories,
        )
