"""Benchmark scenario tables (YAML), loaded by ``fastkernels.workloads``.

This package exists only to ship ``full.yaml`` and ``default.yaml`` as package
data. Edit those files to change which models/workloads are benchmarked; the
loader in ``workloads.py`` turns them into ``BenchmarkScenario`` objects.
"""
