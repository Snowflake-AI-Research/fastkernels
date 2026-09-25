"""Benign-numerics calibration run, for models whose noise run is bit-identical to the baseline.

    python -m fastkernels.e2e.perturb E2E_OUT SCENARIO_SLUG --out DIR

Re-runs the scenario's noise-run spec with ``FASTKERNELS_PERTURB=1`` (adapters that support it
switch to benign numerics; OpenFold3: cuBLAS bf16 reduced-precision reductions off) and writes
``DIR/perturb.json``, scored against the baseline outputs with the same schema as noise.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("e2e_out", type=Path)
    ap.add_argument("slug")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    runs = a.e2e_out / "runs" / a.slug
    spec = json.loads((runs / "noise" / "spec.json").read_text())
    d = a.out / "perturb"
    d.mkdir(parents=True, exist_ok=True)
    spec["result"], spec["run"]["out_dir"] = str(d / "result.json"), str(d)
    (d / "spec.json").write_text(json.dumps(spec, indent=1))
    with open(d / "log.txt", "w") as log:
        subprocess.run([sys.executable, "-m", "fastkernels.e2e.runner", str(d / "spec.json")],
                       env=dict(os.environ, FASTKERNELS_PERTURB="1"), stdout=log, stderr=subprocess.STDOUT)
    res = json.loads((d / "result.json").read_text())
    out = {"model": res.get("model"), "status": res["status"], "error": res.get("error"),
           "perturbation": "FASTKERNELS_PERTURB=1", "timings": res.get("timings")}
    if res["status"] == "ok":
        import torch

        from .adapters import adapter_for
        from ..workloads import resolve_benchmark

        cmp = adapter_for(resolve_benchmark(spec["scenarios"])[spec["index"]]).compare(
            torch.load(runs / "baseline" / "outputs.pt", weights_only=False),
            torch.load(d / "outputs.pt", weights_only=False))
        ps = cmp.get("per_sample") or []
        out.update(correctness=cmp, mean_d=sum(ps) / len(ps) if ps else None)
    (a.out / "perturb.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out.get(k) for k in ("status", "error", "mean_d")}))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
