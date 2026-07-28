#!/usr/bin/env bash
# =============================================================================
# train_lolv2_stage3_pure_low_single.sh
#
# Sequentially run the RetinexPixelTrans / LOLv2 pure_low_single Stage3 local
# search configs around the current Stage2 ExpD basin.
#
# Usage:
#   bash _train/RetinexPixelTrans/pure_low_single/train_lolv2_stage3_pure_low_single.sh
#   bash _train/RetinexPixelTrans/pure_low_single/train_lolv2_stage3_pure_low_single.sh --validate
#
# Optional:
#   RETINEX_PYTHON=/path/to/python bash .../train_lolv2_stage3_pure_low_single.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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
FAILED_LOG="${LOG_DIR}/train_lolv2_stage3_pure_low_single_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_lolv2_stage3_pure_low_single_summary.log"

MODE="${1:-run}"
if [[ "${MODE}" == "--validate" ]]; then
    VALIDATE_ONLY=true
else
    VALIDATE_ONLY=false
fi

TOTAL=9
CURRENT=0
TRAIN_FAILED=0

CONFIGS=(
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpA_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.0125rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpB_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.015rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpC_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpD_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.025rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpE_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.03rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpF_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.05smv4_0.02rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpG_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.075smv4_0.02rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpH_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.15smv4_0.02rtv.yaml"
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpI_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.2smv4_0.02rtv.yaml"
)

LABELS=(
    "ExpA R_TV sweep | rtv 0.0125"
    "ExpB R_TV sweep | rtv 0.015"
    "ExpC R_TV retest | rtv 0.02"
    "ExpD R_TV sweep | rtv 0.025"
    "ExpE R_TV sweep | rtv 0.03"
    "ExpF smooth sweep | sm 0.05 + rtv 0.02"
    "ExpG smooth sweep | sm 0.075 + rtv 0.02"
    "ExpH smooth sweep | sm 0.15 + rtv 0.02"
    "ExpI smooth sweep | sm 0.2 + rtv 0.02"
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
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpA_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.0125rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpB_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.015rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpC_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.02rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpD_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.025rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpE_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.1smv4_0.03rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpF_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.05smv4_0.02rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpG_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.075smv4_0.02rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpH_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.15smv4_0.02rtv.yaml",
    "configs/RetinexPixelTrans/pure_low_single/LOLv2/stage3/Stage3_ExpI_LOLv2_1.0r_0.05anchorv3_0.1bdsp_0.2smv4_0.02rtv.yaml",
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
echo "RetinexPixelTrans LOLv2 pure_low_single Stage3: ${TOTAL} configs"
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
echo "Stage3 pure_low_single training complete"
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
