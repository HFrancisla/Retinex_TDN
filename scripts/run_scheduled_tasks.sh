#!/usr/bin/env bash
# This script runs two training commands sequentially.
# It is scheduled to be run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

cd "${ROOT_DIR}"

echo "========================================="
echo "Starting scheduled tasks sequence"
echo "Date: $(date)"
echo "========================================="

echo "[1/2] Running first task..."
"${PYTHON}" train.py --config configs/RetinexPixelTrans/pure_low_single/LOLv2_stage2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3.yaml

echo "[2/2] Running second task..."
"${PYTHON}" train.py --config configs/RetinexPixelTrans/pure_low_single/LOLv2_stage2_1.0r_0.05anchorv2_0.0bdsp_0.1smv3.yaml

echo "========================================="
echo "Finished scheduled tasks sequence"
echo "Date: $(date)"
echo "========================================="
