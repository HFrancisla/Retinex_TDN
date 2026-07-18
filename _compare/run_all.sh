#!/usr/bin/env bash
# ============================================================
# Retinex 对比分析流水线 — 顺序执行全部 5 个步骤
#
# 用法:
#   bash _compare/run_all.sh                          # 全部执行
#   RETINEX_SYNTH_MAX_ITER=50000 bash _compare/run_all.sh  # 限制迭代范围
#
# 环境变量（可选）:
#   RETINEX_SYNTH_MAX_ITER    最大迭代轮次 (默认 100000)
#   RETINEX_SYNTH_EXP_DIR     实验根目录 (默认 ./experiments)
#   RETINEX_ANALYZE_ITER      分解分析迭代 (默认 10000；设为 all 分析全部)
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
echo "▶ [1/5] synthesize_retinex — 合成 R × L 图像"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/synthesize_retinex.py"
echo "✓ 步骤 1 完成"

# ── Step 2: 计算 PSNR ───────────────────────────────────────
echo ""
echo "▶ [2/5] psnr_synthesis — 计算合成 vs 原始 PSNR"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/psnr_synthesis.py"
echo "✓ 步骤 2 完成"

# ── Step 3: 跨网络对比 HTML ─────────────────────────────────
echo ""
echo "▶ [3/5] generate_compare_html — 生成跨网络对比页面"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/generate_compare_html.py"
echo "✓ 步骤 3 完成"

# ── Step 4: 单网络多 run 对比 HTML ───────────────────────────
echo ""
echo "▶ [4/5] generate_model_compare_html — 生成单网络对比页面"
echo "────────────────────────────────────────────────────────────"
"$PYTHON" "$SCRIPT_DIR/generate_model_compare_html.py"
echo "✓ 步骤 4 完成"

# ── Step 5: R/L 分解诊断分析 ───────────────────────────────
echo ""
echo "▶ [5/5] analyze_decomposition — R/L 分解统计诊断"
echo "────────────────────────────────────────────────────────────"

ANALYZE_ARGS=()
if [ "${RETINEX_FORCE_ANALYZE:-0}" = "1" ]; then
    ANALYZE_ARGS+=(--force)
    echo "  (强制模式：忽略已有报告)"
fi
ANALYZE_ITER="${RETINEX_ANALYZE_ITER:-10000}"
if [ "$ANALYZE_ITER" != "all" ]; then
    ANALYZE_ARGS+=(--iteration "$ANALYZE_ITER")
    echo "  (统一分析 iteration=$ANALYZE_ITER)"
fi

ANALYZE_COUNT=0
ANALYZE_SKIP_COUNT=0
ANALYZE_FAIL_COUNT=0
for model_dir in "$ROOT_DIR/experiments"/*/; do
    [ -d "$model_dir" ] || continue
    for mode_dir in "$model_dir"/*/; do
        [ -d "$mode_dir" ] || continue
        for run_dir in "$mode_dir"/*/; do
            [ -d "$run_dir" ] || continue
            if [ -d "$run_dir/img" ] && [ -f "$run_dir/config.yaml" ]; then
                ANALYZE_COUNT=$((ANALYZE_COUNT + 1))
                rel="${run_dir#$ROOT_DIR/experiments/}"
                echo "[$ANALYZE_COUNT] $rel"
                if output=$("$PYTHON" "$SCRIPT_DIR/analyze_decomposition.py" "${ANALYZE_ARGS[@]}" "$run_dir" 2>&1); then
                    status=0
                else
                    status=$?
                    ANALYZE_FAIL_COUNT=$((ANALYZE_FAIL_COUNT + 1))
                fi
                echo "$output"
                if [ "${status:-0}" -ne 0 ]; then
                    echo "[ERROR] analyze_decomposition failed with status $status: $rel"
                fi
                if echo "$output" | grep -q "\[SKIP\]"; then
                    ANALYZE_SKIP_COUNT=$((ANALYZE_SKIP_COUNT + 1))
                fi
            fi
        done
    done
done

if [ "$ANALYZE_COUNT" -eq 0 ]; then
    echo "⚠ 没有找到包含 img/ 和 config.yaml 的实验目录，跳过。"
else
    if [ "$ANALYZE_SKIP_COUNT" -gt 0 ]; then
        echo "✓ 步骤 5 完成（共 $ANALYZE_COUNT 个实验，跳过 $ANALYZE_SKIP_COUNT 个已有报告，失败 $ANALYZE_FAIL_COUNT 个）"
    else
        echo "✓ 步骤 5 完成（分析了 $ANALYZE_COUNT 个实验）"
    fi
fi

if [ "$ANALYZE_FAIL_COUNT" -gt 0 ]; then
    echo "分析存在 $ANALYZE_FAIL_COUNT 个失败实验；流水线返回非零状态。"
    exit 1
fi

# ── 完成 ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 全部 5 个步骤执行完成。"
echo " 输出文件:"
echo "   experiments/*/synthesis/              — 合成图像"
echo "   experiments/*/synthesis_compare.txt   — PSNR 报告"
echo "   experiments/*/decomposition_analysis.txt — R/L 分解诊断"
echo "   _compare/html/compare.html            — 跨网络对比"
echo "   _compare/html/compare_*.html          — 单网络对比"
echo "============================================================"
