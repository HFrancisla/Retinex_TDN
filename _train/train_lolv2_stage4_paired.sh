#!/usr/bin/env bash
# =============================================================================
# train_lolv2_stage4_paired.sh
#
# Sequentially run the 10 RetinexPixelTrans / LOLv2 paired stage4 configs.
#
# Usage:
#   bash _train/train_lolv2_stage4_paired.sh
#   bash _train/train_lolv2_stage4_paired.sh --validate
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_lolv2_stage4_paired_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_lolv2_stage4_paired_summary.log"

MODE="${1:-run}"
if [[ "${MODE}" == "--validate" ]]; then
    VALIDATE_ONLY=true
else
    VALIDATE_ONLY=false
fi

TOTAL=10
CURRENT=0
TRAIN_FAILED=0

CONFIGS=(
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.6smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.7smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.8smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_1.0smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.5smv2.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.5smv3.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.05eq_0.5smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.15eq_0.5smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.003cross_0.1eq_0.5smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.005cross_0.1eq_0.5smv1.yaml"
)

LABELS=(
    "E1 smooth v1 0.6"
    "E2 smooth v1 0.7"
    "E3 smooth v1 0.8"
    "E4 smooth v1 1.0"
    "E5 smooth v2 0.5"
    "E6 smooth v3 0.5"
    "E7 equal_r 0.05"
    "E8 equal_r 0.15"
    "E9 cross 0.003"
    "E10 cross 0.005"
)

hr() {
    echo "--------------------------------------------------------------------------------"
}

preflight_check() {
    if [[ ! -x "${PYTHON}" ]]; then
        echo "Missing venv python: ${PYTHON}" >&2
        return 1
    fi
    if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
        echo "Missing train script: ${TRAIN_SCRIPT}" >&2
        return 1
    fi

    "${PYTHON}" - <<'PY'
from pathlib import Path
from train import validate_pipeline_config
from utils import load_config
import os
import sys

configs = [
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.6smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.7smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.8smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_1.0smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.5smv2.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.1eq_0.5smv3.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.05eq_0.5smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.001cross_0.15eq_0.5smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.003cross_0.1eq_0.5smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_stage4_0.005cross_0.1eq_0.5smv1.yaml",
]

ok = 0
for cfg_path in configs:
    try:
        path = Path(cfg_path)
        if not path.is_file():
            raise FileNotFoundError(cfg_path)
        cfg = load_config(str(path))
        validate_pipeline_config(cfg)
        data_path = cfg["data"]["path"]
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"data path missing: {data_path}")
        print(f"OK {cfg_path}")
        ok += 1
    except Exception as exc:
        print(f"FAIL {cfg_path}: {exc}", file=sys.stderr)

print(f"Validated {ok}/{len(configs)} configs")
if ok != len(configs):
    sys.exit(1)
PY
}

run_exp() {
    local label="$1"
    local config="$2"

    CURRENT=$((CURRENT + 1))
    echo ""
    hr
    echo "[${CURRENT}/${TOTAL}] ${label}"
    echo "Config: ${config}"
    hr

    if "${PYTHON}" "${TRAIN_SCRIPT}" --config "${ROOT_DIR}/${config}"; then
        echo "PASS [${CURRENT}/${TOTAL}] ${label}" | tee -a "${SUMMARY_LOG}"
    else
        echo "FAIL [${CURRENT}/${TOTAL}] ${label}" | tee -a "${SUMMARY_LOG}"
        echo "${label} | ${config}" >> "${FAILED_LOG}"
        TRAIN_FAILED=$((TRAIN_FAILED + 1))
    fi
}

: > "${FAILED_LOG}"
: > "${SUMMARY_LOG}"

echo ""
echo "RetinexPixelTrans LOLv2 paired stage4: ${TOTAL} configs"
echo "Python: ${PYTHON}"
echo "Root  : ${ROOT_DIR}"
echo ""

preflight_check

if [[ "${VALIDATE_ONLY}" == true ]]; then
    echo "--validate mode: configs are valid; training skipped."
    exit 0
fi

for i in "${!CONFIGS[@]}"; do
    run_exp "${LABELS[$i]}" "${CONFIGS[$i]}"
done

echo ""
hr
echo "Stage4 training complete"
echo "Total : ${TOTAL}"
echo "Pass  : $((TOTAL - TRAIN_FAILED))"
echo "Fail  : ${TRAIN_FAILED}"
echo "Summary log: ${SUMMARY_LOG}"
if [[ "${TRAIN_FAILED}" -gt 0 ]]; then
    echo "Failed log : ${FAILED_LOG}"
    cat "${FAILED_LOG}"
fi
hr

if [[ "${TRAIN_FAILED}" -gt 0 ]]; then
    exit 1
fi
