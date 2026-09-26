"""Run every reference probe in its own process; write <out>/<model>.json, logs and summary.json.

Example:
  python run_all.py --hf-source ~/src/transformers-da6c53e4 --out results/old
  python run_all.py --hf-source ~/src/transformers-89b6b175 --extra-path ~/deps-89b6b175 --out results/new
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probes import PROBES  # noqa: E402

OLD, NEW = "da6c53e431f7c9ef0691239d4ce89b0f711ecad7", "89b6b17574892ec0770551537a3fe69d6886703e"
EXPECTED = {
    OLD: dict(deepseek_v4="wrong_computation", mra="execution_error", reformer="execution_error",
              grounding_dino="execution_error", mm_grounding_dino="execution_error"),
    NEW: dict(deepseek_v4="execution_error", mra="execution_error", reformer="execution_error",
              grounding_dino="wrong_computation", mm_grounding_dino="wrong_computation"),
}
for revision in EXPECTED:
    EXPECTED[revision] = dict(dict(
        clvp="wrong_computation", nllb_moe="wrong_computation", sam_hq="wrong_computation",
        phi4_multimodal="wrong_computation", gemma4_assistant="wrong_computation",
        sam3_video="wrong_computation", granite4_vision="reference_selection_failure",
    ), **EXPECTED[revision])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-source", required=True)
    parser.add_argument("--extra-path", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", choices=sorted(PROBES), default=list(PROBES))
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per probe")
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, start = [], time.perf_counter()
    for model in args.models:
        target = out / f"{model}.json"
        target.unlink(missing_ok=True)
        command = [sys.executable, str(HERE / "probes.py"), model, "--hf-source", args.hf_source,
                   "--out", str(target), "--device", args.device]
        command += [x for p in args.extra_path for x in ("--extra-path", p)]
        began = time.perf_counter()
        with open(out / f"{model}.log", "w") as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ,
                                      timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        record = json.loads(target.read_text()) if target.is_file() else {}
        revision = record.get("transformers_git_revision")
        expected = EXPECTED.get(revision, {}).get(model)
        row = {"model": model, "verdict": record.get("verdict", "no_output"), "expected": expected,
               "matches_expected": None if expected is None else record.get("verdict") == expected,
               "exit_code": code, "seconds": round(time.perf_counter() - began, 1),
               "transformers_git_revision": revision}
        rows.append(row)
        print(f"{model:20s} {row['verdict']:28s} expected={expected} exit={code} {row['seconds']}s", flush=True)
    summary = {"hf_source": args.hf_source, "extra_path": args.extra_path, "device": args.device,
               "total_seconds": round(time.perf_counter() - start, 1), "results": rows,
               "all_match_expected": all(r["matches_expected"] for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary: {out / 'summary.json'}  all_match_expected={summary['all_match_expected']}")
    return 0 if all(r["exit_code"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
