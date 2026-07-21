#!/usr/bin/env bash
# =============================================================================
# train_lolv2_paired_smv1_low.sh
#
# Sequentially rerun the 2 RetinexPixelTrans / LOLv2 paired configs
# with low smv1 values (0.1 and 0.3).
#
# Usage:
#   bash _train/train_lolv2_paired_smv1_low.sh
#   bash _train/train_lolv2_paired_smv1_low.sh --validate
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_lolv2_paired_smv1_low_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_lolv2_paired_smv1_low_summary.log"

MODE="${1:-run}"
if [[ "${MODE}" == "--validate" ]]; then
    VALIDATE_ONLY=true
else
    VALIDATE_ONLY=false
fi

TOTAL=2
CURRENT=0
TRAIN_FAILED=0

CONFIGS=(
    "configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.3smv1.yaml"
)

LABELS=(
    "smv1 0.1"
    "smv1 0.3"
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
    "configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml",
    "configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.3smv1.yaml",
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
echo "RetinexPixelTrans LOLv2 paired rerun smv1 (0.1, 0.3): ${TOTAL} configs"
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
echo "Training complete"
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
