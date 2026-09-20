"""Diff two ``fastkernels validate`` result trees.

Run validate once per vLLM interpreter (``--skip-fastkernels``, distinct
``--run-id``), then:

    python -m fastkernels.validate.compare_vllm_runs vllm-0.18.0 vllm-0.26.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastkernels import RESULTS_DIR

_VALIDATE_ROOT = RESULTS_DIR / "validate"


def _resolve_run(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path.resolve()
    candidate = _VALIDATE_ROOT / name_or_path
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"run {name_or_path!r} not found as a directory or under {_VALIDATE_ROOT}"
    )


def _metric(entry: dict, suffix: str, prefer: str | None = None) -> float | None:
    if prefer and prefer in entry:
        value = entry[prefer]
        return float(value) if isinstance(value, (int, float)) else None
    for key, value in entry.items():
        if key.endswith(suffix) and isinstance(value, (int, float)):
            return float(value)
    return None


def _load_metrics(root: Path) -> dict[tuple[str, str, str], float]:
    """Map (kind, model, workload) -> metric.

    Throughput is tok/s (prefer ``vllm_tok_per_s``); latency is ms/tok
    (prefer ``vllm_ms_per_tok``).
    """
    metrics: dict[tuple[str, str, str], float] = {}
    for path in sorted(root.rglob("results.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        model = data.get("model")
        if not model:
            continue
        for item in data.get("scenarios") or []:
            if not isinstance(item, dict):
                continue
            workload = item.get("scenario") or item.get("name")
            value = _metric(item, "_tok_per_s", "vllm_tok_per_s")
            if workload and value is not None:
                metrics[("throughput", model, workload)] = value
        for item in data.get("latency_scenarios") or []:
            if not isinstance(item, dict):
                continue
            workload = item.get("scenario") or item.get("name")
            value = _metric(item, "_ms_per_tok", "vllm_ms_per_tok")
            if workload and value is not None:
                metrics[("latency", model, workload)] = value
    return metrics


def compare_runs(left: Path, right: Path) -> list[dict]:
    left_metrics = _load_metrics(left)
    right_metrics = _load_metrics(right)
    keys = sorted(set(left_metrics) | set(right_metrics))
    rows: list[dict] = []
    for kind, model, workload in keys:
        a = left_metrics.get((kind, model, workload))
        b = right_metrics.get((kind, model, workload))
        ratio = None
        if a is not None and b is not None and a != 0:
            # Throughput: higher is better (B/A). Latency: lower is better (A/B).
            ratio = (b / a) if kind == "throughput" else (a / b)
        rows.append(
            {
                "kind": kind,
                "model": model,
                "workload": workload,
                "a": a,
                "b": b,
                "speedup": ratio,
            }
        )
    return rows


def _fmt(value: float | None, *, integers: bool = False) -> str:
    if value is None:
        return "—"
    if integers:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _print_table(rows: list[dict], kind: str, a_label: str, b_label: str) -> None:
    subset = [row for row in rows if row["kind"] == kind]
    if not subset:
        return
    if kind == "throughput":
        a_hdr, b_hdr = f"{a_label} tok/s", f"{b_label} tok/s"
        title = "THROUGHPUT"
        integers = True
    else:
        a_hdr, b_hdr = f"{a_label} ms/tok", f"{b_label} ms/tok"
        title = "LATENCY"
        integers = False
    print(f"\n{title}")
    header = (
        f"{'MODEL':<40} {'WORKLOAD':<18} {a_hdr:>14} {b_hdr:>14} {'SPEEDUP':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in subset:
        speedup = (
            f"{row['speedup']:.2f}x" if row["speedup"] is not None else "—"
        )
        print(
            f"{row['model']:<40} {row['workload']:<18} "
            f"{_fmt(row['a'], integers=integers):>14} "
            f"{_fmt(row['b'], integers=integers):>14} {speedup:>8}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two fastkernels validate result trees.",
    )
    parser.add_argument("run_a", help="First run id or directory (baseline).")
    parser.add_argument("run_b", help="Second run id or directory.")
    args = parser.parse_args(argv)
    try:
        left = _resolve_run(args.run_a)
        right = _resolve_run(args.run_b)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = compare_runs(left, right)
    if not rows:
        print("error: no results.json metrics found in either run", file=sys.stderr)
        return 1
    print(f"  A: {left.name}  ({left})")
    print(f"  B: {right.name}  ({right})")
    _print_table(rows, "throughput", left.name, right.name)
    _print_table(rows, "latency", left.name, right.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
