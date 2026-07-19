#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_lolv2_next_pure_single_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_lolv2_next_pure_single_summary.log"

MODE="${1:-run}"
VALIDATE_ONLY=false
if [[ "${MODE}" == "--validate" ]]; then
    VALIDATE_ONLY=true
elif [[ "${MODE}" != "run" ]]; then
    echo "Usage: bash _train/train_lolv2_next_pure_single.sh [--validate]"
    exit 2
fi

CONFIGS=(
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.08anchorv2_0.05bdsp_0.1smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.10anchorv2_0.05bdsp_0.1smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.15anchorv2_0.05bdsp_0.1smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.05anchorv2_0.08bdsp_0.1smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.05anchorv2_0.10bdsp_0.1smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.08anchorv2_0.08bdsp_0.1smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.10anchorv2_0.10bdsp_0.1smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.08anchorv2_0.05bdsp_0.05smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_1.0r_0.08anchorv2_0.05bdsp_0.2smv2.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_next_0.7r_0.10anchorv2_0.08bdsp_0.1smv2.yaml"
)

LABELS=(
    "E01 anchor=0.08 bdsp=0.05 smooth=0.1"
    "E02 anchor=0.10 bdsp=0.05 smooth=0.1"
    "E03 anchor=0.15 bdsp=0.05 smooth=0.1"
    "E04 anchor=0.05 bdsp=0.08 smooth=0.1"
    "E05 anchor=0.05 bdsp=0.10 smooth=0.1"
    "E06 anchor=0.08 bdsp=0.08 smooth=0.1"
    "E07 anchor=0.10 bdsp=0.10 smooth=0.1"
    "E08 anchor=0.08 bdsp=0.05 smooth=0.05"
    "E09 anchor=0.08 bdsp=0.05 smooth=0.2"
    "E10 recon=0.7 anchor=0.10 bdsp=0.08 smooth=0.1"
)

TOTAL="${#CONFIGS[@]}"
CURRENT=0
TRAIN_FAILED=0

validate_configs() {
    "${PYTHON}" - "${CONFIGS[@]}" <<'PY'
import os
import sys
from utils import load_config

ok = 0
for cfg_path in sys.argv[1:]:
    try:
        cfg = load_config(cfg_path)
        data_path = cfg["data"]["path"]
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"data path missing: {data_path}")
        loss = cfg["loss"]
        if loss.get("mode") != "pure_low_single_pixel":
            raise ValueError(f"unexpected loss.mode: {loss.get('mode')}")
        if loss.get("smooth_version") != "v2":
            raise ValueError(f"unexpected smooth_version: {loss.get('smooth_version')}")
        eval_cfg = cfg["training"]["eval"]
        if eval_cfg.get("selection_metric") != "r_low_highref_psnr":
            raise ValueError(
                f"selection_metric should be r_low_highref_psnr: {eval_cfg.get('selection_metric')}"
            )
        print(f"OK {cfg_path}")
        ok += 1
    except Exception as exc:
        print(f"FAIL {cfg_path}: {exc}", file=sys.stderr)

if ok != len(sys.argv) - 1:
    sys.exit(1)
print(f"validated {ok}/{len(sys.argv) - 1} configs")
PY
}

run_exp() {
    local index="$1"
    local label="$2"
    local config="$3"

    CURRENT=$((CURRENT + 1))
    echo ""
    echo "================================================================================"
    echo "  [${CURRENT}/${TOTAL}] ${label}"
    echo "  Config: ${config}"
    echo "================================================================================"

    if "${PYTHON}" "${TRAIN_SCRIPT}" --config "${ROOT_DIR}/${config}"; then
        echo "PASS [${CURRENT}/${TOTAL}] ${label}" | tee -a "${SUMMARY_LOG}"
    else
        echo "FAIL [${CURRENT}/${TOTAL}] ${label}" | tee -a "${SUMMARY_LOG}"
        echo "${index} | ${label} | ${config}" >> "${FAILED_LOG}"
        TRAIN_FAILED=$((TRAIN_FAILED + 1))
    fi
}

: > "${FAILED_LOG}"
: > "${SUMMARY_LOG}"

echo "Validating LOLv2 next pure_low_single configs..."
validate_configs

if [[ "${VALIDATE_ONLY}" == true ]]; then
    echo "Validation passed. Training skipped."
    exit 0
fi

echo ""
echo "Starting ${TOTAL} sequential RetinexPixelTrans LOLv2 pure_low_single runs."
echo "Logs:"
echo "  summary: ${SUMMARY_LOG}"
echo "  failed : ${FAILED_LOG}"

for i in "${!CONFIGS[@]}"; do
    run_exp "$((i + 1))" "${LABELS[$i]}" "${CONFIGS[$i]}"
done

echo ""
echo "Completed ${TOTAL} runs: passed=$((TOTAL - TRAIN_FAILED)), failed=${TRAIN_FAILED}"
echo "Summary log: ${SUMMARY_LOG}"
if [[ "${TRAIN_FAILED}" -gt 0 ]]; then
    echo "Failed log: ${FAILED_LOG}"
    exit 1
fi
