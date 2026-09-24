"""Summaries and checks for ``fastkernels e2e`` output directories.

    python -m fastkernels.e2e.report OUT                    # status table
    python -m fastkernels.e2e.report OUT --check-preflight  # exit 1 unless every model passed
    python -m fastkernels.e2e.report --verify-sets DIR      # candidate-set checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def _geo(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def _fmt(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(out: Path) -> list[dict]:
    rows = []
    for sdir in sorted((out / "results").glob("*")):
        if not sdir.is_dir():
            continue
        for f in sorted(sdir.glob("*.json")):
            r = json.loads(f.read_text())
            rows.append({"scenario": sdir.name, "item": f.stem, "status": r.get("status"),
                         "requested": len(r.get("kernels_requested") or []),
                         "final": None if r.get("kernels_final") is None else len(r["kernels_final"]),
                         "dropped": [(d["kernel"], d["category"]) for d in r.get("dropped") or []],
                         "all_winners": (r.get("all_winners") or {}).get("status"),
                         "speedup": _geo(list((r.get("speedups") or {}).values())),
                         "mean_d": r.get("mean_d"),
                         "error": (r.get("error") or "")[:120]})
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"{'scenario':42} {'item':13} {'status':16} {'req':>4} {'fin':>4} {'allwin':>7} "
          f"{'speedup':>8} {'mean_d':>7}  dropped / error")
    for r in rows:
        extra = ", ".join(f"{k}({c})" for k, c in r["dropped"]) or r["error"]
        print(f"{r['scenario'][:42]:42} {r['item']:13} {str(r['status']):16} {r['requested']:>4} "
              f"{'-' if r['final'] is None else r['final']:>4} {str(r['all_winners'] or '-'):>7} "
              f"{_fmt(r['speedup']):>8} {_fmt(r['mean_d']):>7}  {extra[:90]}")


def check_preflight(rows: list[dict], expected: int | None) -> bool:
    ok = True
    scen = {r["scenario"] for r in rows}
    if expected is not None and len(scen) < expected:
        print(f"PREFLIGHT: only {len(scen)}/{expected} scenarios produced results")
        ok = False
    for r in rows:
        if r["item"] == "baseline" and r["status"] != "ok":
            print(f"PREFLIGHT FAIL {r['scenario']}: baseline {r['status']} {r['error']}")
            ok = False
        if r["item"] == "selftest" and (r["status"] != "ok" or r["dropped"]
                                         or (r["mean_d"] is not None and r["mean_d"] > 0.2)):
            print(f"PREFLIGHT FAIL {r['scenario']}: selftest status={r['status']} "
                  f"dropped={r['dropped']} mean_d={r['mean_d']}")
            ok = False
    print("PREFLIGHT " + ("PASSED" if ok else "FAILED"))
    return ok


def verify_sets(root: Path) -> bool:
    ok = True
    for manifest in sorted(root.glob("*/manifest.json")):
        m = json.loads(manifest.read_text())
        bad = [rel for rel, h in (m.get("files") or {}).items()
               if not (manifest.parent / rel).is_file()
               or hashlib.sha256((manifest.parent / rel).read_bytes()).hexdigest() != h]
        print(f"{manifest.parent.name:14} {len(m.get('files') or {}):4} files  "
              f"{'OK' if not bad else 'MISMATCH ' + ', '.join(bad[:5])}")
        ok &= not bad
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--check-preflight", action="store_true")
    ap.add_argument("--expected-scenarios", type=int, default=None)
    ap.add_argument("--verify-sets", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.verify_sets:
        return 0 if verify_sets(args.verify_sets) else 1
    rows = summarize(args.out)
    print_table(rows)
    if args.check_preflight:
        return 0 if check_preflight(rows, args.expected_scenarios) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
