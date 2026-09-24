"""One e2e run in a fresh process: ``python -m fastkernels.e2e.runner SPEC.json``.

SPEC.json::

    {"scenarios": "default", "index": 0,          # which scenario
     "run": {...RunSpec fields...},
     "candidates": true | false,                   # swap in the candidate set?
     "result": "/path/result.json"}

The candidate set, and which of its kernels to use, come from the environment, which must
be set before this process starts (``fastkernels`` reads it at import):
``FASTKERNELS_CANDIDATE_DIR``, ``FASTKERNELS_CANDIDATE_ONLY``,
``FASTKERNELS_CANDIDATE_EXCLUDE``. The result JSON is always written, also on a crash.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _apply_candidates(scenario) -> dict:
    from fastkernels.list import apply_candidates, discover_candidate_impls
    if scenario.tp > 1:
        # Spawned tensor-parallel workers are fresh interpreters: re-apply the swap there
        # through eval's env-gated sitecustomize (they inherit ONLY/EXCLUDE as well).
        from fastkernels.eval import _APPLY_CANDIDATES_ENV, _install_candidate_sitecustomize
        _install_candidate_sitecustomize()
        os.environ[_APPLY_CANDIDATES_ENV] = "1"
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        pairs = discover_candidate_impls()  # imports (and JIT-builds) the candidates
    t_import = time.time() - t0
    sys.stdout.write(buf.getvalue())
    apply_candidates(pairs)
    swapped = [f"L{t.level}:{t.name}" for t, _, _ in pairs]
    requested = [k for k in os.environ.get("FASTKERNELS_CANDIDATE_ONLY", "").split(",") if k]
    return {
        "swapped": swapped,
        # Requested kernels whose candidate failed to import or define its class.
        "not_swapped": [k for k in requested if k not in swapped],
        "import_messages": [l for l in buf.getvalue().splitlines() if "skip candidate" in l],
        "t_import_s": round(t_import, 1),
    }


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[0]).read_text())
    result: dict = {"status": "crash", "candidates": bool(spec.get("candidates")),
                    "candidate_dir": os.environ.get("FASTKERNELS_CANDIDATE_DIR")
                    if spec.get("candidates") else None}
    t_start = time.time()
    try:
        from fastkernels.e2e.adapters import RunSpec, adapter_for
        from fastkernels.workloads import resolve_benchmark

        scenario = resolve_benchmark(spec["scenarios"])[spec["index"]]
        adapter_cls = adapter_for(scenario)
        result.update(model=scenario.hf_name, tp=scenario.tp, adapter=adapter_cls.name)
        run = RunSpec(**spec["run"])
        Path(run.out_dir).mkdir(parents=True, exist_ok=True)
        if spec.get("candidates"):
            result.update(_apply_candidates(scenario))
        t0 = time.time()
        result["timings"] = adapter_cls().run(scenario, run)
        result["t_run_s"] = round(time.time() - t0, 1)
        result["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001 -- every failure is a result
        result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        result["traceback"] = traceback.format_exc()[-30000:]
        traceback.print_exc()
    result["t_total_s"] = round(time.time() - t_start, 1)
    Path(spec["result"]).parent.mkdir(parents=True, exist_ok=True)
    Path(spec["result"]).write_text(json.dumps(result, indent=1))
    sys.stdout.flush()
    # os._exit: engines leave non-daemon threads / NCCL groups that can hang interpreter exit.
    os._exit(0 if result["status"] == "ok" else 3)


if __name__ == "__main__":
    main(sys.argv[1:])
