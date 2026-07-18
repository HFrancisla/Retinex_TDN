#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

CONFIGS=(
  "configs/RetinexPixelTrans/paired/LOLv2_stage2_0.01cross_0.1eq_0.1smv2.yaml"
  "configs/RetinexPixelTrans/paired/LOLv2_stage2_0.05cross_0.1eq_0.1smv2.yaml"
  "configs/RetinexPixelTrans/paired/LOLv2_stage2_0.001cross_0.2eq_0.1smv2.yaml"
  "configs/RetinexPixelTrans/paired/LOLv2_stage2_0.01cross_0.2eq_0.1smv2.yaml"
  "configs/RetinexPixelTrans/paired/LOLv2_stage2_0.01cross_0.2eq_0.1smv3.yaml"
)

cd "${ROOT_DIR}"

echo "[stage2] Python: ${PYTHON}"
echo "[stage2] Sequential experiments: ${#CONFIGS[@]}"

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  number=$((index + 1))
  echo
  echo "[stage2] (${number}/${#CONFIGS[@]}) START ${config}"
  "${PYTHON}" train.py --config "${config}"
  echo "[stage2] (${number}/${#CONFIGS[@]}) DONE  ${config}"
done

echo
echo "[stage2] All experiments completed."
