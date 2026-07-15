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
from typing import Any

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
