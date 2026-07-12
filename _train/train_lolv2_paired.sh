#!/usr/bin/env bash
# =============================================================================
# train_lolv2_paired.sh — LOLv2 paired：4 个网络，共 6 次
#
# 使用 .venv 中的 Python 环境，依次执行 6 组 paired 训练。
# RetinexPixelTrans 包含 3 个 smooth_weight 变体（0.1 / 0.3 / 0.5）。
#
# 用法:
#   bash _train/train_lolv2_paired.sh                # 直接训练（默认）
#   bash _train/train_lolv2_paired.sh --validate     # 仅预检: 配置验证 + 冒烟测试
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"
SMOKE_SCRIPT="${ROOT_DIR}/_train/smoke_test.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_lolv2_paired_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_lolv2_paired_summary.log"

MODE="${1:-run}"
if [[ "$MODE" == "--validate" ]]; then
    SKIP_CHECK=false;  VALIDATE_ONLY=true
else
    SKIP_CHECK=true;   VALIDATE_ONLY=false
fi

TOTAL=6
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
    echo "  ║          预检验证 — LOLv2 paired (6 配置)                            ║"
    echo "  ╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    # ---- 阶段 1：配置 + 数据路径 ----
    echo -e "${CYAN}  ▸ 阶段 1/2：验证 6 个配置文件与数据路径${NC}"
    hr

    "${PYTHON}" -c "
from utils import load_config
import os, sys

configs = [
    ('RetinexPointRaw       | paired              | LOLv2', 'configs/RetinexPointRaw/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq.yaml'),
    ('RetinexPixelClassic    | paired              | LOLv2', 'configs/RetinexPixelClassic/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml'),
    ('RetinexPixelTrans      | paired sm=0.1       | LOLv2', 'configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml'),
    ('RetinexPixelTrans      | paired sm=0.3       | LOLv2', 'configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.3smv1.yaml'),
    ('RetinexPixelTrans      | paired sm=0.5       | LOLv2', 'configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.5smv1.yaml'),
    ('RetinexPixelTransMinus | paired              | LOLv2', 'configs/RetinexPixelTransMinus/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml'),
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
    echo -e "${CYAN}  ▸ 阶段 2/2：冒烟测试 — 每类别跑 5 个 training step${NC}"
    echo "     (验证模型/损失/数据三者能正确对接)"
    hr

    "${PYTHON}" "${SMOKE_SCRIPT}" --subset lolv2_paired || { failed=1; }

    hr
    if [[ $failed -ne 0 ]]; then
        echo -e "  ${RED}冒烟测试失败，终止。使用 --skip-check 可跳过预检强制训练。${NC}"
        return 1
    fi
    echo -e "  ${GREEN}预检全部通过 ✅  可以开始 LOLv2 paired 训练${NC}"
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
#  LOLv2 paired — 4 网络 × 1 + RetinexPixelTrans × 2 (sm 变体) = 6 次
# =============================================================================
echo ""
echo "  ████████████████████████████████████████████████████████████████████████"
echo "  █  LOLv2 paired：4 个网络 + 2 个 smooth 变体，共 6 次训练"
echo "  ████████████████████████████████████████████████████████████████████████"

run_exp "RetinexPointRaw | LOLv2 paired" \
    "configs/RetinexPointRaw/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq.yaml"

run_exp "RetinexPixelClassic | LOLv2 paired" \
    "configs/RetinexPixelClassic/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml"

run_exp "RetinexPixelTrans sm=0.1 | LOLv2 paired" \
    "configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml"

run_exp "RetinexPixelTrans sm=0.3 | LOLv2 paired" \
    "configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.3smv1.yaml"

run_exp "RetinexPixelTrans sm=0.5 | LOLv2 paired" \
    "configs/RetinexPixelTrans/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.5smv1.yaml"

run_exp "RetinexPixelTransMinus | LOLv2 paired" \
    "configs/RetinexPixelTransMinus/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml"

# =============================================================================
#  汇总
# =============================================================================
echo ""
echo "  ╔══════════════════════════════════════════════════════════════════════╗"
echo "  ║              LOLv2 paired 训练全部完成                               ║"
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
