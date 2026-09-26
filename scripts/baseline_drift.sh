#!/usr/bin/env bash
# Baseline drift check: re-benchmark the pinned production references against a newer vLLM
# release on this node and compute per-family drift coefficients (fastkernels.validate.drift).
#
#   bash scripts/baseline_drift.sh                 # candidate = latest vLLM on PyPI
#   CANDIDATE=0.29.0 bash scripts/baseline_drift.sh
#
# Exits 0 when the candidate is the pin or was already checked on this GPU, 2 when a model
# family drifted by more than THRESHOLD (a FastKernels re-release is due), 1 on errors.
# Run by .github/workflows/baseline-drift.yml on a self-hosted GPU runner.
#
# Measurement hygiene (from the vLLM 0.18/0.26/0.29 ablation): every version gets its own
# isolated venv without vllm-omni (its plugin auto-registers into every vLLM process), jobs
# run strictly serially with the whole node (FASTKERNELS_VALIDATE_SERIAL=1), and vLLM's usage
# thread is disabled (its stdout parsing trips the harness watchdog).
set -euo pipefail
cd "$(dirname "$0")/.."

PINNED=$(sed -n 's/.*"vllm==\([0-9.]*\)".*/\1/p' pyproject.toml | head -1)
CANDIDATE=${CANDIDATE:-$(curl -fsS https://pypi.org/pypi/vllm/json | python -c "import sys,json; print(json.load(sys.stdin)['info']['version'])")}
SCENARIOS=${SCENARIOS:-vllm_only}
THRESHOLD=${THRESHOLD:-0.10}
NGPU=${NGPU:-8}
VENV_ROOT=${FK_VENV_ROOT:-$HOME/.fastkernels/drift-venvs}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cs 'A-Za-z0-9' '-' | sed 's/-$//')
OUT=${DRIFT_OUT:-drift-report}/$GPU/vllm-$PINNED-to-$CANDIDATE
DAY=$(date +%Y%m%d)  # fixed once: a sweep can cross midnight
export FASTKERNELS_VALIDATE_SERIAL=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

echo "[drift] gpu=$GPU pinned=vllm-$PINNED candidate=vllm-$CANDIDATE scenarios=$SCENARIOS"
if [ "$CANDIDATE" = "$PINNED" ]; then echo "[drift] candidate is the pinned release; nothing to do"; exit 0; fi
if [ -f "$OUT/drift.json" ]; then echo "[drift] already checked: $OUT/drift.md"; cat "$OUT/drift.md"; exit 0; fi

venv_for() {  # <version> -> python of an isolated venv with that vLLM release
  local v=$1 py="$VENV_ROOT/vllm-$1/bin/python"
  if [ ! -x "$py" ]; then
    uv venv -q --python 3.12 "$VENV_ROOT/vllm-$v" >&2
    uv pip install -q --python "$py" --torch-backend auto "vllm==$v" fastsafetensors ninja datasets av \
      opencv-python-headless pillow >&2
    local fiv; fiv=$("$py" -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
    # Prebuilt flashinfer kernels (not on PyPI); without them MoE models JIT-compile at run time.
    uv pip install -q --python "$py" --no-deps --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==$fiv" >&2 || true
    uv pip install -q --python "$py" --no-deps --index-url https://flashinfer.ai/whl "flashinfer-cubin==$fiv" >&2 || true
  fi
  "$py" - "$NGPU" <<'PY' >&2
import importlib.metadata as m, sys, torch, vllm
print(f"[drift]   vllm {vllm.__version__} torch {torch.__version__} transformers {m.version('transformers')}")
try:
    m.version("vllm-omni"); sys.exit("[drift]   vllm-omni present: this venv would contaminate the measurement")
except m.PackageNotFoundError:
    pass
if torch.cuda.device_count() < int(sys.argv[1]):
    sys.exit(f"[drift]   expected {sys.argv[1]} GPUs, see {torch.cuda.device_count()}")
PY
  echo "$py"
}

for v in "$PINNED" "$CANDIDATE"; do
  py=$(venv_for "$v")
  # --skip-fastkernels benchmarks only the references; validate then exits non-zero by design.
  python -m fastkernels validate "$SCENARIOS" --skip-fastkernels --vllm-python "$py" \
    --run-id "drift-$GPU-vllm-$v-$DAY" --timeout 18000 || true
done

mkdir -p "$OUT"
python -m fastkernels.validate.drift "drift-$GPU-vllm-$PINNED-$DAY" "drift-$GPU-vllm-$CANDIDATE-$DAY" --threshold "$THRESHOLD" --out "$OUT"
