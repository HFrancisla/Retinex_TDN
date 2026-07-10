#!/usr/bin/env bash
# ============================================================
# Retinex 对比分析流水线 — 顺序执行全部 4 个步骤
#
# 用法:
#   bash _compare/run_all.sh                          # 全部执行
#   RETINEX_SYNTH_MAX_ITER=50000 bash _compare/run_all.sh  # 限制迭代范围
#
# 环境变量（可选）:
#   RETINEX_SYNTH_MAX_ITER    最大迭代轮次 (默认 100000)
#   RETINEX_SYNTH_EXP_DIR     实验根目录 (默认 ./experiments)
#   PYTHON_BIN                Python 解释器 (默认 .venv/bin/python3，回退 python3)
# ============================================================

set -euo pipefail

# ── 确定项目根目录 ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# ── 查找 Python 解释器 ──────────────────────────────────────
PYTHON="${PYTHON_BIN:-}"
if [ -z "$PYTHON" ]; then
    if [ -f "$ROOT_DIR/.venv/bin/python3" ]; then
        PYTHON="$ROOT_DIR/.venv/bin/python3"
    else
        PYTHON="python3"
    fi
fi

echo "============================================================"
echo " Retinex 对比分析流水线"
echo " Python  : $PYTHON"
echo " 项目目录: $ROOT_DIR"
echo "============================================================"

# ── Step 1: 合成 S = R × L ──────────────────────────────────
echo ""
echo "▶ [1/4] synthesize_retinex — 合成 R × L 图像"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/synthesize_retinex.py"
echo "✓ 步骤 1 完成"

# ── Step 2: 计算 PSNR ───────────────────────────────────────
echo ""
echo "▶ [2/4] psnr_synthesis — 计算合成 vs 原始 PSNR"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/psnr_synthesis.py"
echo "✓ 步骤 2 完成"

# ── Step 3: 跨网络对比 HTML ─────────────────────────────────
echo ""
echo "▶ [3/4] generate_compare_html — 生成跨网络对比页面"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/generate_compare_html.py"
echo "✓ 步骤 3 完成"

# ── Step 4: 单网络多 run 对比 HTML ───────────────────────────
echo ""
echo "▶ [4/4] generate_model_compare_html — 生成单网络对比页面"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/generate_model_compare_html.py"
echo "✓ 步骤 4 完成"

# ── 完成 ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 全部 4 个步骤执行完成。"
echo " 输出文件:"
echo "   experiments/*/synthesis/         — 合成图像"
echo "   experiments/*/synthesis_compare.txt — PSNR 报告"
echo "   _compare/html/compare.html       — 跨网络对比"
echo "   _compare/html/compare_*.html     — 单网络对比"
echo "============================================================"
