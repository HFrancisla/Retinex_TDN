#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"

echo "=========================================================================="
echo "Verifying smv2 configuration (E5)"
echo "=========================================================================="
"${PYTHON}" "${TRAIN_SCRIPT}" --config "${ROOT_DIR}/configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.5smv2.yaml"

echo "=========================================================================="
echo "Verifying smv3 configuration (E6)"
echo "=========================================================================="
"${PYTHON}" "${TRAIN_SCRIPT}" --config "${ROOT_DIR}/configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.5smv3.yaml"

echo "Verification training complete."
