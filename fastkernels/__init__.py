"""fastkernels: a CUDA kernel benchmarking library.

Canonical path resolution lives here so every other module can
``from fastkernels import KB_ROOT, CANDIDATE_DIR, ...`` instead of
computing paths via ``Path(__file__)``.

Override any path with an environment variable for CI, Docker, or
non-standard layouts.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Package root: the fastkernels/ directory itself.
KB_ROOT = Path(os.environ.get("FASTKERNELS_ROOT", str(Path(__file__).resolve().parent)))

# One level up from fastkernels/ (the repo checkout directory).
PROJECT_ROOT = KB_ROOT.parent

# --- Task directories ---
TASKS_DIR = KB_ROOT / "tasks"
BASELINE_DIR = TASKS_DIR / "baseline"
CANDIDATE_DIR = Path(
    os.environ.get("FASTKERNELS_CANDIDATE_DIR", str(TASKS_DIR / "candidate"))
)
PREV_ATTEMPTS_DIR = CANDIDATE_DIR / "prev-attempts"

# --- Benchmark results ---
RESULTS_DIR = Path(
    os.environ.get("FASTKERNELS_RESULTS_DIR", str(KB_ROOT / "bench" / "results"))
)

# --- MLflow tracking ---
MLFLOW_TRACKING_DIR = KB_ROOT / "mlruns"

# --- Agent build cache ---
CUDA_BUILD_CACHE = KB_ROOT / "agent" / "_cuda_build_cache"


def run_output_path(tool: str, ext: str = "json") -> Path:
    """Return a timestamped output path, e.g. ``bench/results/kernels_20260313_143022.json``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{tool}_{ts}.{ext}"
