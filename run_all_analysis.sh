#!/bin/bash
# 一键运行所有的对比与分析流程

# 确保在项目根目录运行
cd "$(dirname "$0")" || exit 1

FORCE_ARGS=""
if [[ "$1" == "--force" ]]; then
    echo "⚠️ 检测到 --force 参数，所有分析将强制忽略缓存重新生成 ⚠️"
    FORCE_ARGS="--force"
    export RETINEX_SYNTH_FORCE=1
    export RETINEX_FORCE_ANALYZE=1
fi

echo "=========================================="
echo "1. 正在运行 _compare/run_all.sh ..."
echo "=========================================="
if bash _compare/run_all.sh; then
    echo "=> _compare/run_all.sh 运行成功！"
    echo ""
    
    echo "=========================================="
    echo "2. 正在运行 paired 分析脚本 ..."
    echo "=========================================="
    if bash _analysis/RetinexPixelTrans/paired/run_paired.sh $FORCE_ARGS; then
        echo "=> paired 分析脚本执行完毕！"
    else
        echo "=> [错误] paired 分析脚本执行失败！"
        exit 1
    fi
    echo ""
    
    echo "=========================================="
    echo "3. 正在运行 pure_low_single 分析脚本 ..."
    echo "=========================================="
    if bash _analysis/RetinexPixelTrans/pure_low_single/run_pure_low_single.sh $FORCE_ARGS; then
        echo "=> pure_low_single 分析脚本执行完毕！"
    else
        echo "=> [错误] pure_low_single 分析脚本执行失败！"
        exit 1
    fi
    echo ""
    
    echo "=========================================="
    echo "所有分析流程已成功执行完毕！"
    echo "=========================================="
else
    echo "=> [错误] _compare/run_all.sh 运行失败，已中止后续分析流程！"
    exit 1
fi
