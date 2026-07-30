#!/usr/bin/env bash
# =============================================================================
# train_best_loss_all_models.sh
#
# Sequentially run the BestLoss configurations across different models
# for LOLv2 and BDDnight.
#
# Usage:
#   bash _train/train_best_loss_all_models.sh
#   bash _train/train_best_loss_all_models.sh --validate
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${RETINEX_PYTHON:-}" ]]; then
    PYTHON="${RETINEX_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON="$(command -v python3 || true)"
fi
TRAIN_SCRIPT="${ROOT_DIR}/train.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_best_loss_all_models_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_best_loss_all_models_summary.log"

MODE="${1:-run}"
if [[ "${MODE}" == "--validate" ]]; then
    VALIDATE_ONLY=true
else
    VALIDATE_ONLY=false
fi

TOTAL=4
CURRENT=0
TRAIN_FAILED=0

CONFIGS=(
    "configs/RetinexPixelTransMinus/pure_low_single/LOLv2/BestLoss_RetinexPixelTransMinus.yaml"
    "configs/RetinexPixelClassic/pure_low_single/LOLv2/BestLoss_RetinexPixelClassic.yaml"
    "configs/RetinexPointRaw/pure_low_single/LOLv2/BestLoss_RetinexPointRaw.yaml"
    "configs/RetinexPixelTransMinus/pure_low_single/BDD/BestLoss_RetinexPixelTransMinus_BDD.yaml"
)

LABELS=(
    "LOLv2 | RetinexPixelTransMinus"
    "LOLv2 | RetinexPixelClassic"
    "LOLv2 | RetinexPointRaw"
    "BDDnight | RetinexPixelTransMinus"
)

hr() {
    echo "--------------------------------------------------------------------------------"
}

preflight_check() {
    if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
        echo "Missing python executable. Set RETINEX_PYTHON=/path/to/python." >&2
        return 1
    fi
    if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
        echo "Missing train script: ${TRAIN_SCRIPT}" >&2
        return 1
    fi

    (
        cd "${ROOT_DIR}"
        "${PYTHON}" - <<'PY'
from pathlib import Path
from train import generate_experiment_name, validate_pipeline_config
from utils import load_config
import os
import sys

configs = [
    "configs/RetinexPixelTransMinus/pure_low_single/LOLv2/BestLoss_RetinexPixelTransMinus.yaml",
    "configs/RetinexPixelClassic/pure_low_single/LOLv2/BestLoss_RetinexPixelClassic.yaml",
    "configs/RetinexPointRaw/pure_low_single/LOLv2/BestLoss_RetinexPointRaw.yaml",
    "configs/RetinexPixelTransMinus/pure_low_single/BDD/BestLoss_RetinexPixelTransMinus_BDD.yaml",
]

names = []
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
        exp_name = generate_experiment_name(cfg)
        names.append(exp_name)
        print(f"OK {cfg_path}")
        print(f"   auto_name -> {exp_name}")
        ok += 1
    except Exception as exc:
        print(f"FAIL {cfg_path}: {exc}", file=sys.stderr)

if len(names) != len(set(names)):
    print("FAIL duplicate generated experiment names", file=sys.stderr)
    for name in names:
        print(f"  {name}", file=sys.stderr)
    sys.exit(1)

print(f"Validated {ok}/{len(configs)} configs")
if ok != len(configs):
    sys.exit(1)
PY
    )
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
echo "BestLoss validation and training: ${TOTAL} configs"
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
