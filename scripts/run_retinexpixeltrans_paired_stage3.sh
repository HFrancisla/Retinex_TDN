#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"

CONFIGS=(
  "configs/RetinexPixelTrans/paired/LOLv2_stage3_0.05cross_0.2eq_0.1smv3.yaml"
  "configs/RetinexPixelTrans/paired/LOLv2_stage3_0.1cross_0.2eq_0.1smv3.yaml"
  "configs/RetinexPixelTrans/paired/LOLv2_stage3_0.05cross_0.3eq_0.1smv3.yaml"
  "configs/RetinexPixelTrans/paired/LOLv2_stage3_0.05cross_0.2eq_0.2smv3.yaml"
)

cd "${ROOT_DIR}"

echo "[stage3] Python: ${PYTHON}"
echo "[stage3] Sequential experiments: ${#CONFIGS[@]}"

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  number=$((index + 1))
  echo
  echo "[stage3] (${number}/${#CONFIGS[@]}) START ${config}"
  "${PYTHON}" train.py --config "${config}"
  echo "[stage3] (${number}/${#CONFIGS[@]}) DONE  ${config}"
done

echo
echo "[stage3] All experiments completed."
