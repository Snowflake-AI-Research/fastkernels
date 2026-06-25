"""CLI for ``fastkernels history`` and ``fastkernels mlflow-ui``.

Queries the local MLflow tracking store and prints human-readable
tables.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime


def _fmt_time(ts) -> str:
    """Format a timestamp (ms-epoch or datetime) to a short string."""
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M")
        return str(ts)[:16]
    except Exception:
        return str(ts)[:16]


def _short_id(run_id: str) -> str:
    return run_id[:8] if run_id else ""


# ------------------------------------------------------------------
# fastkernels history
# ------------------------------------------------------------------
def history_main():
    """Entry point for ``fastkernels history``."""
    parser = argparse.ArgumentParser(
        prog="fastkernels history",
        description="Query tracked experiment runs from MLflow.",
    )
    parser.add_argument(
        "--op", type=str, default=None,
        help="Show history for a specific operator (e.g. 'rms_norm').",
    )
    parser.add_argument(
        "--best", action="store_true",
        help="Show best speedup per operator (kernel benchmarks only).",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of runs to display (default: 20).",
    )
    args = parser.parse_args()

    from fastkernels.bench.tracking.tracker import _ensure_init

    if not _ensure_init():
        print("  mlflow is not installed. Install with: pip install mlflow")
        return

    if args.best:
        _print_best(args.limit)
    elif args.op:
        _print_operator_history(args.op, args.limit)
    else:
        _print_recent(args.limit)


def _print_recent(limit: int) -> None:
    from fastkernels.bench.tracking.tracker import query_runs

    runs = query_runs(max_results=limit)
    if not runs:
        print("  No tracked runs found.")
        return

    print(f"\n{'=' * 70}")
    print("  RECENT TRACKED RUNS")
    print(f"{'=' * 70}")
    print(
        f"  {'TIMESTAMP':<18} {'RUN NAME':<35} {'KEY METRICS'}"
    )
    print(f"  {'\u2500' * 66}")

    for r in runs:
        ts = _fmt_time(r.get("start_time", ""))
        name = r.get("tags.mlflow.runName", r.get("run_id", "")[:8])
        if len(name) > 33:
            name = name[:30] + "..."

        # Extract key metrics for summary
        parts = []
        for key in (
            "metrics.avg_speedup",
            "metrics.total_passed",
            "metrics.total_failed",
            "metrics.e2e_speedup",
            "metrics.e2e_token_match_rate",
            "metrics.avg_throughput_speedup",
            "metrics.tokens_per_second",
            "metrics.avg_latency",
        ):
            val = r.get(key)
            if val is not None and val == val:  # skip NaN
                short_key = key.split(".")[-1]
                try:
                    if "speedup" in short_key:
                        parts.append(f"{short_key}={val:.2f}x")
                    elif "rate" in short_key:
                        parts.append(f"{short_key}={val:.1%}")
                    elif "passed" in short_key or "failed" in short_key:
                        parts.append(f"{short_key}={int(val)}")
                    else:
                        parts.append(f"{short_key}={val:.1f}")
                except (ValueError, TypeError):
                    pass

        summary = "  ".join(parts[:3]) if parts else "--"
        print(f"  {ts:<18} {name:<35} {summary}")

    print(f"{'=' * 70}")


def _print_operator_history(op_name: str, limit: int) -> None:
    from fastkernels.bench.tracking.tracker import query_runs

    runs = query_runs(max_results=limit * 3)  # fetch extra, then filter

    # Filter to runs that have metrics for this operator
    speedup_key = f"metrics.{op_name}_avg_speedup"
    gen_key = f"metrics.gen_{op_name}_success"

    matching = [
        r for r in runs
        if r.get(speedup_key) is not None or r.get(gen_key) is not None
    ]

    if not matching:
        print(f"  No tracked data found for operator '{op_name}'.")
        return

    print(f"\n{'=' * 70}")
    print(f"  TRACKING HISTORY: {op_name}")
    print(f"{'=' * 70}")
    print(
        f"  {'TIMESTAMP':<18} {'RUN NAME':<28} "
        f"{'SPEEDUP':>8} {'PASS':>6} {'ERR_RATIO':>10} {'RUN ID':>10}"
    )
    print(f"  {'\u2500' * 66}")

    for r in matching[:limit]:
        ts = _fmt_time(r.get("start_time", ""))
        name = r.get("tags.mlflow.runName", "")[:26]
        run_id = _short_id(r.get("run_id", ""))

        speedup = r.get(speedup_key)
        passed = r.get(f"metrics.{op_name}_passed")
        failed = r.get(f"metrics.{op_name}_failed")
        error_ratio = r.get(f"metrics.{op_name}_avg_max_error_ratio")
        gen = r.get(gen_key)

        speedup_s = f"{speedup:.2f}x" if speedup is not None and speedup == speedup else "--"
        if (
            passed is not None and passed == passed
            and failed is not None and failed == failed
        ):
            pass_s = f"{int(passed)}/{int(passed + failed)}"
        elif gen is not None and gen == gen:
            pass_s = "gen=OK" if gen else "gen=FAIL"
        else:
            pass_s = "--"
        ratio_s = (
            f"{error_ratio:.2e}"
            if error_ratio is not None and error_ratio == error_ratio else "--"
        )

        print(
            f"  {ts:<18} {name:<28} "
            f"{speedup_s:>8} {pass_s:>6} {ratio_s:>10} {run_id:>10}"
        )

    print(f"{'=' * 70}")


def _print_best(limit: int) -> None:
    from fastkernels.bench.tracking.tracker import query_runs

    runs = query_runs(max_results=500)
    if not runs:
        print("  No tracked runs found.")
        return

    # Collect best speedup per operator
    best: dict[str, tuple[float, str, str]] = {}  # op -> (speedup, date, run_id)
    for r in runs:
        for key, val in r.items():
            if (
                key.startswith("metrics.")
                and key.endswith("_avg_speedup")
                and val is not None
                and val == val  # skip NaN
            ):
                op = key[len("metrics."):-len("_avg_speedup")]
                if op in ("", "avg"):
                    continue
                ts = _fmt_time(r.get("start_time", ""))
                run_id = _short_id(r.get("run_id", ""))
                if op not in best or val > best[op][0]:
                    best[op] = (val, ts, run_id)

    if not best:
        print("  No kernel benchmark data found.")
        return

    print(f"\n{'=' * 70}")
    print("  BEST SPEEDUP PER OPERATOR (from kernel benchmarks)")
    print(f"{'=' * 70}")
    print(
        f"  {'OPERATOR':<25} {'BEST SPEEDUP':>13} "
        f"{'DATE':<18} {'RUN ID':>10}"
    )
    print(f"  {'\u2500' * 66}")

    for op in sorted(best):
        speedup, ts, run_id = best[op]
        print(f"  {op:<25} {speedup:>12.2f}x {ts:<18} {run_id:>10}")

    print(f"{'=' * 70}")


# ------------------------------------------------------------------
# fastkernels mlflow-ui
# ------------------------------------------------------------------
def mlflow_ui_main():
    """Entry point for ``fastkernels mlflow-ui``."""
    from fastkernels import MLFLOW_TRACKING_DIR

    backend_store = f"file://{MLFLOW_TRACKING_DIR}"
    print("  Starting MLflow UI...")
    print(f"  Tracking store: {MLFLOW_TRACKING_DIR}")
    print("  Open http://localhost:5000 in your browser")
    print("  Press Ctrl+C to stop")

    try:
        subprocess.run(
            [
                sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", backend_store,
                "--port", "5000",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print("  ERROR: mlflow is not installed. Install with: pip install mlflow")
