"""MLflow tracking facade for fastkernels.

All MLflow interaction is isolated in this module.  Other fastkernels code
imports ``tracker`` and calls its functions; no other module should
import ``mlflow`` directly.

If mlflow is not installed, every public function silently becomes a
no-op after printing a single warning.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastkernels.bench.eval.aggregator import EvalReport
    from fastkernels.bench.kernels.result import KernelBenchResult

# ---------------------------------------------------------------------------
# Lazy mlflow handle
# ---------------------------------------------------------------------------
_mlflow: Any = None
_initialized: bool = False
_warned: bool = False


def _ensure_init() -> bool:
    """Lazy-init: import mlflow and set tracking URI on first call.

    Returns True if mlflow is available.
    """
    global _mlflow, _initialized, _warned
    if _initialized:
        return _mlflow is not None

    _initialized = True
    try:
        import mlflow

        _mlflow = mlflow
    except ImportError:
        if not _warned:
            _warned = True
            warnings.warn(
                "mlflow is not installed — experiment tracking is disabled. "
                "Install with: pip install 'fastkernels[tracking]'",
                stacklevel=3,
            )
        return False

    from fastkernels import MLFLOW_TRACKING_DIR

    tracking_uri = f"file://{MLFLOW_TRACKING_DIR}"
    _mlflow.set_tracking_uri(tracking_uri)
    return True


def _safe(fn):
    """Decorator: swallow exceptions from MLflow so logging never crashes."""

    def wrapper(*args, **kwargs):
        if not _ensure_init():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"MLflow logging error (ignored): {exc}", stacklevel=2)
            return None

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------
@contextmanager
def start_run(
    name: str,
    params: dict[str, Any] | None = None,
    experiment: str = "fastkernels",
    tags: dict[str, str] | None = None,
):
    """Open an MLflow run.  Use as a context manager.

    Parameters
    ----------
    name : str
        Human-readable run name (e.g. ``"agent_L1_llama"``).
    params : dict, optional
        Run parameters to log (model, level, tp, …).
    experiment : str
        MLflow experiment name (default ``"fastkernels"``).
    tags : dict, optional
        Extra MLflow tags.

    Yields
    ------
    The ``mlflow.ActiveRun`` if MLflow is available, otherwise ``None``.
    """
    if not _ensure_init():
        yield None
        return

    try:
        _mlflow.set_experiment(experiment)
        with _mlflow.start_run(run_name=name) as run:
            if params:
                # MLflow params must be strings — convert non-str values
                safe_params = {
                    k: str(v) for k, v in params.items() if v is not None
                }
                _mlflow.log_params(safe_params)
            if tags:
                _mlflow.set_tags(tags)
            yield run
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"MLflow run error (ignored): {exc}", stacklevel=2)
        yield None


# ---------------------------------------------------------------------------
# Kernel logging
# ---------------------------------------------------------------------------
@_safe
def log_kernel(
    op_name: str,
    level: int,
    code: str,
    error: str | None = None,
) -> None:
    """Log a generated kernel.

    Stores the source code as an MLflow artifact under
    ``kernels/{op_name}.py`` and records success/failure as a metric.
    """
    success = error is None and bool(code)
    _mlflow.log_metric(f"gen_{op_name}_success", int(success))

    if code:
        _mlflow.log_text(code, f"kernels/{op_name}.py")
    if error:
        _mlflow.log_text(error[:4000], f"errors/{op_name}.txt")


# ---------------------------------------------------------------------------
# Benchmark logging — kernel tier
# ---------------------------------------------------------------------------
@_safe
def log_kernel_bench(result: KernelBenchResult) -> None:
    """Log a ``KernelBenchResult`` (from ``fastkernels kernels``).

    Logs per-operator and per-scenario metrics, plus aggregate totals.
    Candidate kernel source files are stored as artifacts.
    """
    from fastkernels import CANDIDATE_DIR

    for op in result.operators:
        _mlflow.log_metric(f"{op.target}_avg_speedup", op.avg_speedup)
        _mlflow.log_metric(f"{op.target}_passed", op.passed)
        _mlflow.log_metric(f"{op.target}_failed", op.failed)
        _mlflow.log_metric(
            f"{op.target}_avg_max_error_ratio", op.avg_max_error_ratio
        )
        _mlflow.log_metric(
            f"{op.target}_avg_mean_abs_diff", op.avg_mean_abs_diff
        )
        for s in op.scenarios:
            key = f"{op.target}.{s.name}"
            # Truncate metric keys to 250 chars (MLflow limit)
            if len(key) > 240:
                key = key[:240]
            _mlflow.log_metric(f"{key}_speedup", s.speedup)
            _mlflow.log_metric(f"{key}_correct", int(s.correct))
            _mlflow.log_metric(f"{key}_max_error_ratio", s.max_error_ratio)

        # Store candidate kernel source as artifact
        candidate_file = CANDIDATE_DIR / f"L{op.level}" / f"{op.target}.py"
        if candidate_file.is_file():
            _mlflow.log_artifact(str(candidate_file), artifact_path="kernels")

    # Aggregates
    _mlflow.log_metric("avg_speedup", result.avg_speedup)
    _mlflow.log_metric("total_passed", result.passed)
    _mlflow.log_metric("total_failed", result.failed)
    _mlflow.log_metric("total_operators", result.total_operators)
    _mlflow.log_metric("total_scenarios", result.total_scenarios)
    _mlflow.log_metric("avg_max_error_ratio", result.avg_max_error_ratio)
    _mlflow.log_metric("avg_mean_abs_diff", result.avg_mean_abs_diff)


# ---------------------------------------------------------------------------
# Benchmark logging — eval tier
# ---------------------------------------------------------------------------
@_safe
def log_eval(report: EvalReport) -> None:
    """Log an ``EvalReport`` (from ``fastkernels eval``)."""
    for cat in report.categories:
        for m in cat.models:
            if m.status == "FAILED":
                continue
            short = m.model.split("/")[-1]
            prefix = f"{short}_tp{m.tp}"
            _mlflow.log_metric(
                f"{prefix}_throughput_speedup", m.throughput_speedup
            )
            _mlflow.log_metric(
                f"{prefix}_latency_speedup", m.latency_speedup
            )
            _mlflow.log_metric(
                f"{prefix}_alignment_rate", m.alignment_rate
            )

    _mlflow.log_metric(
        "avg_throughput_speedup", report.avg_throughput_speedup
    )
    _mlflow.log_metric("avg_latency_speedup", report.avg_latency_speedup)
    _mlflow.log_metric("alignment_rate", report.alignment_rate)
    _mlflow.log_metric("macro_speedup", report.macro_speedup)
    _mlflow.log_metric("macro_correctness", report.macro_correctness)
    _mlflow.log_metric("macro_coverage", report.macro_coverage)
    _mlflow.log_metric("macro_score", report.macro_score)
    _mlflow.log_metric("wall_clock_seconds", report.wall_clock_seconds)
    _mlflow.log_metric("failed_jobs", report.failed_jobs)


# ---------------------------------------------------------------------------
# Benchmark logging — e2e tier
# ---------------------------------------------------------------------------
@_safe
def log_e2e(results: dict, bench_type: str) -> None:
    """Log E2E benchmark results.

    Parameters
    ----------
    results : dict
        The result dictionary produced by throughput/latency/serve benchmarks.
    bench_type : str
        One of ``"throughput"``, ``"latency"``, ``"serve"``.
    """
    _mlflow.set_tag("bench_type", bench_type)

    if bench_type == "throughput":
        for key in (
            "tokens_per_second",
            "output_tokens_per_second",
            "requests_per_second",
            "elapsed_time",
            "total_input_tokens",
            "total_output_tokens",
        ):
            if key in results:
                _mlflow.log_metric(key, results[key])

    elif bench_type == "latency":
        if "avg_latency" in results:
            _mlflow.log_metric("avg_latency", results["avg_latency"])
        percentiles = results.get("percentiles", {})
        for p, val in percentiles.items():
            _mlflow.log_metric(f"p{p}_latency", val)

    elif bench_type == "serve":
        for key in (
            "mean_ttft_ms",
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "p99_itl_ms",
            "mean_e2el_ms",
            "p99_e2el_ms",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
        ):
            val = results.get(key)
            if val is None and hasattr(results, key):
                val = getattr(results, key)
            if val is not None:
                _mlflow.log_metric(key, val)


# ---------------------------------------------------------------------------
# Custom metrics
# ---------------------------------------------------------------------------
@_safe
def log_metrics(metrics: dict[str, float]) -> None:
    """Log arbitrary key-value metrics to the active run."""
    _mlflow.log_metrics(metrics)


# ---------------------------------------------------------------------------
# Query helpers (for ``fastkernels history``)
# ---------------------------------------------------------------------------
def query_runs(
    experiment: str = "fastkernels",
    filter_string: str | None = None,
    max_results: int = 50,
) -> list[dict]:
    """Search MLflow runs.  Returns a list of dicts (one per run).

    Each dict has keys like ``run_id``, ``run_name``, ``start_time``,
    ``params.*``, ``metrics.*``.
    """
    if not _ensure_init():
        return []

    try:
        exp = _mlflow.get_experiment_by_name(experiment)
        if exp is None:
            return []

        df = _mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string or "",
            max_results=max_results,
            order_by=["start_time DESC"],
        )
        if df.empty:
            return []
        return df.to_dict("records")
    except Exception:  # noqa: BLE001
        return []
