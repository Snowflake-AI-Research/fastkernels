#!/usr/bin/env bash
# One-shot setup for the isolated sglang benchmark environment.
#
# We keep sglang in its own conda env (`sglang-bench`) so that its preferred
# torch / CUDA versions (currently torch 2.9.1 + cu128) do not conflict with
# the main `dev` env (torch 2.10 + cu130) that fastkernels runs in.
#
# Usage:
#   bash tests/setup_sglang_env.sh
#
# Then point the bench script at it (this is already the default):
#   python tests/bench_sglang.py \
#       --sglang-python "$(conda info --base)/envs/sglang-bench/bin/python"

set -euo pipefail

ENV_NAME="${ENV_NAME:-sglang-bench}"
PY_VERSION="${PY_VERSION:-3.12}"

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
ENV_PY="${ENV_PREFIX}/bin/python"

if [[ ! -x "${ENV_PY}" ]]; then
    echo "[setup] Creating conda env '${ENV_NAME}' (python ${PY_VERSION})..."
    conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
else
    echo "[setup] Reusing existing conda env: ${ENV_PREFIX}"
fi

echo "[setup] Installing pip + uv into '${ENV_NAME}'..."
"${ENV_PY}" -m pip install --upgrade pip uv

echo "[setup] Installing sglang (default cu128 wheel) into '${ENV_NAME}'..."
"${ENV_PY}" -m uv pip install --python "${ENV_PY}" sglang

echo "[setup] Smoke-test:"
"${ENV_PY}" - <<'PY'
import sglang as sgl
import torch
print(f"  sglang : {sgl.__version__}")
print(f"  torch  : {torch.__version__} (cuda {torch.version.cuda})")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
PY

echo "[setup] Done. Use:"
echo "    python tests/bench_sglang.py --sglang-python '${ENV_PY}'"
