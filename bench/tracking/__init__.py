"""MLflow experiment tracking for kb_nano benchmarks.

Provides a simple API for tracking kernel generation, benchmarks, and
optimization progress across runs.  All MLflow interaction is contained
in ``tracker.py``; other kb_nano modules import only this package.

Quick start::

    from kb_nano.bench.tracking import tracker

    with tracker.start_run("my-run", params={"model": "llama", "level": 1}):
        tracker.log_kernel("rms_norm", level=1, code=src)
        tracker.log_kernel_bench(bench_result)
        tracker.log_metrics({"custom_score": 0.95})
"""

from kb_nano.bench.tracking import tracker

__all__ = ["tracker"]
