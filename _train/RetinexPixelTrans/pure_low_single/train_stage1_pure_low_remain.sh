#!/usr/bin/env bash
# =============================================================================
# train_stage1_pure_low_remain.sh — 训练剩下的 5 个纯低光分解（Stage1）的配置
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"
SMOKE_SCRIPT="${ROOT_DIR}/_train/smoke_test.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_stage1_pure_low_remain_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_stage1_pure_low_remain_summary.log"

MODE="${1:-run}"
if [[ "$MODE" == "--validate" ]]; then
    SKIP_CHECK=false;  VALIDATE_ONLY=true
else
    SKIP_CHECK=true;   VALIDATE_ONLY=false
fi

TOTAL=5
CURRENT=0
TRAIN_FAILED=0

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

hr() { echo "────────────────────────────────────────────────────────────────────────────────"; }

# =============================================================================
#  预检
# =============================================================================
preflight_check() {
    local failed=0

    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════════════╗"
    echo "  ║       预检验证 — Stage1 pure_low_single (剩下的 5 配置)               ║"
    echo "  ╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    # ---- 阶段 1：配置 + 数据路径 ----
    echo -e "${CYAN}  ▸ 阶段 1/2：验证 5 个配置文件与数据路径${NC}"
    hr

    "${PYTHON}" -c "
from utils import load_config
import os, sys

configs = [
    ('Trans | 1.0r a2(0.08) b0.08 sm0.1v2', 'configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_next_1.0r_0.08anchorv2_0.08bdsp_0.1smv2.yaml'),
    ('Trans | 1.0r a2(0.10) b0.05 sm0.1v2', 'configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_next_1.0r_0.10anchorv2_0.05bdsp_0.1smv2.yaml'),
    ('Trans | 1.0r a2(0.10) b0.10 sm0.1v2', 'configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_next_1.0r_0.10anchorv2_0.10bdsp_0.1smv2.yaml'),
    ('Trans | stage2 1.0r a2 b0.05 sm0.1v2', 'configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_stage2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2.yaml'),
    ('Trans | stage2 1.0r a2 b0.05 sm0.1v3', 'configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_stage2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3.yaml')
]

ok = 0
for label, cfg_path in configs:
    try:
        cfg = load_config(cfg_path)
        dpath = cfg['data']['path']
        assert os.path.isdir(dpath), f'data path missing: {dpath}'
        print(f'  \033[32m✅\033[0m {label}')
        ok += 1
    except Exception as e:
        print(f'  \033[31m❌\033[0m {label}  ({e})')

print(f'\n  结果: {ok}/{len(configs)} 通过')
if ok != len(configs):
    sys.exit(1)
" || { failed=1; }

    hr
    if [[ $failed -ne 0 ]]; then
        echo -e "  ${RED}阶段 1 失败，终止。${NC}"
        return 1
    fi

    # ---- 阶段 2：冒烟测试 ----
    echo ""
    echo -e "${CYAN}  ▸ 阶段 2/2：冒烟测试 — 挑选部分跑 5 个 training step${NC}"
    echo "     (验证模型/损失/数据三者能正确对接)"
    hr

    "${PYTHON}" "${SMOKE_SCRIPT}" --subset lolv2_pure_single || { failed=1; }

    hr
    if [[ $failed -ne 0 ]]; then
        return 1
    fi
    echo -e "  ${GREEN}预检全部通过 ✅  可以开始 Stage1 pure_low_single 训练${NC}"
    echo ""
}

# ---- 训练执行函数 ----
run_exp() {
    local label="$1"
    local config="$2"

    CURRENT=$((CURRENT + 1))
    echo ""
    echo "================================================================================"
    echo "  [${CURRENT}/${TOTAL}]  ${label}"
    echo "  Config: ${config}"
    echo "================================================================================"
    echo ""

    if "${PYTHON}" "${TRAIN_SCRIPT}" --config "${config}"; then
        echo "  ✅  PASS  [${CURRENT}/${TOTAL}]  ${label}" | tee -a "${SUMMARY_LOG}"
    else
        echo "  ❌  FAIL  [${CURRENT}/${TOTAL}]  ${label}" | tee -a "${SUMMARY_LOG}"
        echo "  ${label}  |  ${config}" >> "${FAILED_LOG}"
        TRAIN_FAILED=$((TRAIN_FAILED + 1))
    fi
}

# =============================================================================
#  主流程
# =============================================================================

:> "${FAILED_LOG}"
:> "${SUMMARY_LOG}"

# ---- 预检 ----
if [[ "$SKIP_CHECK" != true ]]; then
    preflight_check || exit 1
fi

if [[ "$VALIDATE_ONLY" == true ]]; then
    echo "  --validate 模式：预检通过，跳过训练。"
    exit 0
fi

# =============================================================================
#  Stage1 剩下的 5 次训练
# =============================================================================
echo ""
echo "  ████████████████████████████████████████████████████████████████████████"
echo "  █  Stage1 pure_low_single： 剩下的 5 次消融实验训练"
echo "  ████████████████████████████████████████████████████████████████████████"

run_exp "Stage1 | 1.0r a2(0.08) b0.08 sm0.1v2" \
    "configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_next_1.0r_0.08anchorv2_0.08bdsp_0.1smv2.yaml"

run_exp "Stage1 | 1.0r a2(0.10) b0.05 sm0.1v2" \
    "configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_next_1.0r_0.10anchorv2_0.05bdsp_0.1smv2.yaml"
run_exp "Stage1 | 1.0r a2(0.10) b0.10 sm0.1v2" \
    "configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_next_1.0r_0.10anchorv2_0.10bdsp_0.1smv2.yaml"

run_exp "Stage1 | stage2 1.0r a2 b0.05 sm0.1v2" \
    "configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_stage2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2.yaml"
run_exp "Stage1 | stage2 1.0r a2 b0.05 sm0.1v3" \
    "configs/RetinexPixelTrans/pure_low_single/Stage1_LOLv2_stage2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3.yaml"

# =============================================================================
#  汇总
# =============================================================================
echo ""
echo "  ╔══════════════════════════════════════════════════════════════════════╗"
echo "  ║           Stage1 pure_low_single 训练全部完成                       ║"
echo "  ╠══════════════════════════════════════════════════════════════════════╣"
printf "  ║  总计: %-4s  通过: %-4s  失败: %-4s                              ║\n" \
    "${TOTAL}" "$((TOTAL - TRAIN_FAILED))" "${TRAIN_FAILED}"
echo "  ╚══════════════════════════════════════════════════════════════════════╝"
echo ""

if [ "${TRAIN_FAILED}" -gt 0 ]; then
    echo "  失败列表:"
    cat "${FAILED_LOG}"
    echo ""
    echo "  完整日志: ${FAILED_LOG}"
fi
echo "  汇总日志: ${SUMMARY_LOG}"
echo ""
