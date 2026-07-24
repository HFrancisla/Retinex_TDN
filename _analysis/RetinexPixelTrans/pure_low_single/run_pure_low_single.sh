#!/bin/bash
# 运行 pure_low_single 模式的所有分析步骤

# 确保以项目根目录作为当前工作目录
cd "$(dirname "$0")/../../.." || exit 1

PYTHON_EXEC=".venv/bin/python"

if [ ! -x "$PYTHON_EXEC" ]; then
    echo "错误: 未找到虚拟环境中的 Python，请检查 $PYTHON_EXEC 是否存在。"
    exit 1
fi

"$PYTHON_EXEC" _analysis/RetinexPixelTrans/pure_low_single/steps/99_run_all_pure_low_single_steps.py "$@"
