"""Per-family drift coefficients between two ``fastkernels validate`` runs of the references.

Run the reference harness once per reference revision (``validate <scenarios>
--skip-fastkernels --vllm-python <venv>/bin/python --run-id <id>``), then:

    python -m fastkernels.validate.drift <pinned-run> <candidate-run> [--threshold 0.10] [--out DIR]

For every model, the throughput coefficient is the geometric mean over its throughput workloads
of candidate/pinned tok/s, and the latency coefficient the geometric mean over its latency
workloads of pinned/candidate ms/tok (>1: the newer revision is faster). Family coefficients
weight the family's models equally. A speedup measured against the pinned baseline divided by
the family coefficient estimates the speedup against the candidate revision.

Exit code 2 when any family coefficient moves by more than ``--threshold`` (a re-release is due),
0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from .compare_vllm_runs import _resolve_run, compare_runs

# Model families of the reference scenarios (fastkernels/scenarios/vllm_only.yaml).
FAMILIES = {
    "LLM (dense + MoE)": ("Llama-3.1-8B", "Mixtral-8x7B", "gpt-oss", "GLM-5.2", "gemma-4"),
    "Linear attention / SSM / hybrid": ("mamba-2.8b", "Mamba-Codestral", "Qwen3-Next", "Kimi-Linear",
                                        "Jamba"),
    "Audio": ("whisper",),
    "Multimodal / VLM": ("Qwen2-VL", "Qwen3-VL", "Qwen2.5-Omni"),
}


def family_of(model: str) -> str:
    for fam, keys in FAMILIES.items():
        if any(k.lower() in model.lower() for k in keys):
            return fam
    return "Other"


def _geo(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def drift(pinned: Path, candidate: Path) -> dict:
    per_model: dict[str, dict[str, list[float]]] = {}
    for row in compare_runs(pinned, candidate):
        if row["speedup"] is not None:
            per_model.setdefault(row["model"], {"throughput": [], "latency": []})[row["kind"]].append(row["speedup"])
    models = {m: {"family": family_of(m), "throughput": _geo(v["throughput"]), "latency": _geo(v["latency"])}
              for m, v in sorted(per_model.items())}
    families = {}
    for fam in sorted({v["family"] for v in models.values()}):
        ms = [v for v in models.values() if v["family"] == fam]
        families[fam] = {"n": len(ms), "throughput": _geo([v["throughput"] for v in ms]),
                         "latency": _geo([v["latency"] for v in ms])}
    median = {k: statistics.median([v[k] for v in models.values() if v[k]]) if models else None
              for k in ("throughput", "latency")}
    return {"pinned": str(pinned), "candidate": str(candidate), "models": models, "families": families,
            "median_model": median}


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}x"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pinned", help="validate run id or directory of the pinned reference revision")
    ap.add_argument("candidate", help="validate run id or directory of the newer reference revision")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="re-release when a family coefficient moves by more than this (default 0.10)")
    ap.add_argument("--out", type=Path, default=None, help="write drift.json and drift.md here")
    a = ap.parse_args(argv)

    res = drift(_resolve_run(a.pinned), _resolve_run(a.candidate))
    over = [f for f, v in res["families"].items()
            if any(c and max(c, 1 / c) - 1 > a.threshold for c in (v["throughput"], v["latency"]))]
    res.update(threshold=a.threshold, families_over_threshold=over, rerelease_due=bool(over))

    lines = [f"# Reference drift: {Path(res['pinned']).name} -> {Path(res['candidate']).name}", "",
             "| family | n | throughput | latency |", "| --- | --- | --- | --- |"]
    lines += [f"| {f} | {v['n']} | {_fmt(v['throughput'])} | {_fmt(v['latency'])} |" for f, v in res["families"].items()]
    lines += [f"| median model | {len(res['models'])} | {_fmt(res['median_model']['throughput'])} | "
              f"{_fmt(res['median_model']['latency'])} |", "",
              f"Re-release due (a family moved by more than {a.threshold:.0%}): "
              + (", ".join(over) if over else "no")]
    report = "\n".join(lines)
    print(report)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "drift.json").write_text(json.dumps(res, indent=1))
        (a.out / "drift.md").write_text(report + "\n")
    return 2 if over else 0


if __name__ == "__main__":
    sys.exit(main())
