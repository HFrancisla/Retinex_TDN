#!/usr/bin/env bash
# =============================================================================
# train_lolv2_ablation.sh — LOLv2 PixelTrans 消融 A–D，共 4 次
#
# 对 RetinexPixelTrans 在 LOLv2 pure_low_single 模式下做消融实验，
# 变化 recon_weight 与 smooth_weight 两个超参：
#   A (基线)     : recon=1.0  smooth=0.1
#   B (强平滑)   : recon=1.0  smooth=0.5
#   C (弱重建)   : recon=0.3  smooth=0.1
#   D (推荐优先) : recon=0.3  smooth=0.5
#
# 用法:
#   bash scripts/train_lolv2_ablation.sh                # 直接训练（默认）
    #   bash #   bash scripts/train_lolv2_ablation.sh --validate     # 仅预检: 配置验证 + 冒烟测试（不训练） --validate     # 仅预检: 配置验证 + 冒烟测试
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${ROOT_DIR}/train.py"
SMOKE_SCRIPT="${ROOT_DIR}/scripts/smoke_test.py"

LOG_DIR="${ROOT_DIR}/_tmp"
mkdir -p "${LOG_DIR}"
FAILED_LOG="${LOG_DIR}/train_lolv2_ablation_failed.log"
SUMMARY_LOG="${LOG_DIR}/train_lolv2_ablation_summary.log"

MODE="${1:-run}"
if [[ "$MODE" == "--validate" ]]; then
    SKIP_CHECK=false;  VALIDATE_ONLY=true
else
    SKIP_CHECK=true;   VALIDATE_ONLY=false
fi

TOTAL=4
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
    echo "  ║        预检验证 — LOLv2 PixelTrans 消融 A–D (4 配置)                ║"
    echo "  ╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    # ---- 阶段 1：配置 + 数据路径 ----
    echo -e "${CYAN}  ▸ 阶段 1/2：验证 4 个配置文件与数据路径${NC}"
    hr

    "${PYTHON}" -c "
from utils import load_config
import os, sys

configs = [
    ('RetinexPixelTrans | ablation A (r=1.0 sm=0.1) | LOLv2', 'configs/RetinexPixelTrans/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1sm.yaml'),
    ('RetinexPixelTrans | ablation B (r=1.0 sm=0.5) | LOLv2', 'configs/RetinexPixelTrans/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.5sm.yaml'),
    ('RetinexPixelTrans | ablation C (r=0.3 sm=0.1) | LOLv2', 'configs/RetinexPixelTrans/pure_low_single/LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1sm.yaml'),
    ('RetinexPixelTrans | ablation D (r=0.3 sm=0.5) | LOLv2', 'configs/RetinexPixelTrans/pure_low_single/LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.5sm.yaml'),
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

    "${PYTHON}" "${SMOKE_SCRIPT}" --subset lolv2_ablation || { failed=1; }

    hr
    if [[ $failed -ne 0 ]]; then
        return 1
    fi
    echo -e "  ${GREEN}预检全部通过 ✅  可以开始 LOLv2 PixelTrans 消融训练${NC}"
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
#  LOLv2 RetinexPixelTrans 消融 A–D — 4 次
# =============================================================================
echo ""
echo "  ████████████████████████████████████████████████████████████████████████"
echo "  █  LOLv2 RetinexPixelTrans 消融 A–D：4 次训练"
echo "  █  recon_weight ∈ {1.0, 0.3}  ✕  smooth_weight ∈ {0.1, 0.5}"
echo "  ████████████████████████████████████████████████████████████████████████"

run_exp "消融 A (基线)     | LOLv2 recon=1.0 smooth=0.1" \
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1sm.yaml"
run_exp "消融 B (强平滑)   | LOLv2 recon=1.0 smooth=0.5" \
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.5sm.yaml"
run_exp "消融 C (弱重建)   | LOLv2 recon=0.3 smooth=0.1" \
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1sm.yaml"
run_exp "消融 D (推荐优先) | LOLv2 recon=0.3 smooth=0.5" \
    "configs/RetinexPixelTrans/pure_low_single/LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.5sm.yaml"

# =============================================================================
#  汇总
# =============================================================================
echo ""
echo "  ╔══════════════════════════════════════════════════════════════════════╗"
echo "  ║           LOLv2 PixelTrans 消融训练全部完成                          ║"
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
