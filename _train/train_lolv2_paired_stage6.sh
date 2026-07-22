#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd \"$(dirname \"$0\")/..\" && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_lolv2_paired_stage6_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_lolv2_paired_stage6_summary.log"

TOTAL=6
CURRENT=0
TRAIN_FAILED=0

CONFIGS=(
    "configs/RetinexPixelTrans/paired/LOLv2_stage6_1.0rh_0.3rl_0.001crh_0.001crl_0.15eq_0.6smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage6_1.0rh_0.3rl_0.001crh_0.001crl_0.18eq_0.6smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage6_1.0rh_0.3rl_0.001crh_0.001crl_0.2eq_0.6smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage6_0.8rh_0.3rl_0.001crh_0.001crl_0.15eq_0.6smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage6_0.8rh_0.3rl_0.001crh_0.001crl_0.18eq_0.6smv1.yaml"
    "configs/RetinexPixelTrans/paired/LOLv2_stage6_0.8rh_0.3rl_0.001crh_0.001crl_0.2eq_0.6smv1.yaml"
)

LABELS=(
    "rh 1.0 er 0.15"
    "rh 1.0 er 0.18"
    "rh 1.0 er 0.2"
    "rh 0.8 er 0.15"
    "rh 0.8 er 0.18"
    "rh 0.8 er 0.2"
)

hr() {
    echo "--------------------------------------------------------------------------------"
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

echo "RetinexPixelTrans LOLv2 paired stage 6 (rh 1.0, 0.8 / er 0.15, 0.18, 0.20): ${TOTAL} configs"
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
    exit 1
fi
