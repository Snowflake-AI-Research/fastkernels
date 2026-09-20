#!/usr/bin/env bash
# Isolated uv venvs for vLLM 0.18 / 0.29, then validate + compare.
# Current python (fastkernels / vLLM 0.26) is left untouched.
#
#   bash tests/compare_vllm_versions.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

HOST_PY="$(command -v python)"
echo "[setup] host python (0.26 / fastkernels): ${HOST_PY}"
"${HOST_PY}" - <<'PY'
import importlib.metadata as m
import vllm, fastkernels
print(f"  vllm       : {vllm.__version__}")
print(f"  fastkernels: {m.version('fastkernels')}")
PY

uv python install 3.12 >/dev/null

install_one() {
  local ver="$1"
  local extra="${2:-}"
  local py=".venv-vllm-${ver}/bin/python"

  echo "[setup] venv .venv-vllm-${ver}"
  uv venv --python 3.12 ".venv-vllm-${ver}"
  # shellcheck disable=SC2086
  uv pip install --python "${py}" ${extra} \
    "vllm==${ver}" fastsafetensors ninja datasets av opencv-python-headless pillow
  FI="$("${py}" -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")"
  uv pip install --python "${py}" "flashinfer-cubin==${FI}" || true
}

install_one 0.18.0
install_one 0.29.0 "--torch-backend auto"

run_one() {
  local ver="$1"
  local py="$2"
  echo "[run] vllm ${ver}"
  # skip-fastkernels makes validate exit 1 even when jobs PASS (no FK speedup).
  python -m fastkernels validate vllm_only \
    --skip-fastkernels \
    --vllm-python "${py}" \
    --run-id "vllm-${ver}-vllm-only" \
    --timeout 18000 || true
}

run_one 0.18.0 ".venv-vllm-0.18.0/bin/python"
run_one 0.26.0 "${HOST_PY}"
run_one 0.29.0 ".venv-vllm-0.29.0/bin/python"

python -m fastkernels.validate.compare_vllm_runs vllm-0.18.0-vllm-only vllm-0.26.0-vllm-only
python -m fastkernels.validate.compare_vllm_runs vllm-0.26.0-vllm-only vllm-0.29.0-vllm-only
