#!/usr/bin/env bash
# FastKernels paper: end-to-end evaluation of the agents' kernel sets on one 8-GPU node.
#
#   bash scripts/paper_e2e.sh
#
# Re-run the same command to resume after any interruption (finished work is skipped).
# When it prints DONE (or PREFLIGHT FAILED), email back the .tgz file it names (a few MB).
#
# Needs: this repo installed (`pip install -e .`), a Hugging Face login with access to the
# default models (Llama-3.1, FLUX.1-dev, ...), git access to the candidates repo below.
set -euo pipefail

OUT=${OUT:-$HOME/fk-paper-e2e}
CANDIDATES_REPO=${CANDIDATES_REPO:-git@github.com:sfc-gh-goliaro/fastkernels-results.git}
CANDIDATES_REF=${CANDIDATES_REF:-paper-e2e-v1}   # tag pinning the frozen candidate sets
SETS="drkernel-ind drkernel-seq claude-ind claude-seq kda-ind kda-seq ako-ind ako-seq"

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[paper-e2e] $(date) fastkernels=$(git rev-parse --short HEAD) out=$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

pack() {  # always leave a small tarball to email back, even on failure
  (cd "$OUT" && tar czf paper_e2e_results.tgz --ignore-failed-read run.log \
      preflight/results preflight/progress.jsonl e2e/results e2e/progress.jsonl 2>/dev/null) || true
  echo "[paper-e2e] results file: $OUT/paper_e2e_results.tgz"
}
trap pack EXIT

# 1. Frozen candidate sets, checksum-verified.
if [ ! -d "$OUT/candidates/.git" ]; then
  git clone -q --depth 1 --branch "$CANDIDATES_REF" "$CANDIDATES_REPO" "$OUT/candidates"
fi
C="$OUT/candidates/agent-candidates"
python -m fastkernels.e2e.report --verify-sets "$C"

# 2. Preflight (~30-60 min): every model with tiny workloads and a do-nothing candidate set.
python -m fastkernels e2e default --sets "$C/selftest" --out "$OUT/preflight" \
  --max-requests 2 --probe-requests 2 --correctness-samples 2 || true
if ! python -m fastkernels.e2e.report "$OUT/preflight" --check-preflight --expected-scenarios 11; then
  echo "[paper-e2e] PREFLIGHT FAILED -- please email back the results file below"
  exit 1
fi

# 3. Full run (~1-2 days): pre-build each set's kernels once, then every model x set.
SET_DIRS=$(for s in $SETS; do printf "%s," "$C/$s"; done)
python -m fastkernels e2e default --sets "${SET_DIRS%,}" --out "$OUT/e2e" --prebuild
python -m fastkernels.e2e.report "$OUT/e2e"
echo "[paper-e2e] DONE -- please email back the results file below"
