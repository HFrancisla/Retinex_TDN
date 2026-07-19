# RetinexPixelTrans paired 实验分析报告

生成日期：2026-07-19  
分析对象：`experiments/RetinexPixelTrans/paired/*`  
分析轮次：`iteration=10000`

## 执行流程

已按 `_analysis/RetinexPixelTrans/paired/steps/README.md` 的流程完成：

```bash
RETINEX_ANALYZE_ITER=10000 bash _compare/run_all.sh
.venv/bin/python _analysis/RetinexPixelTrans/paired/steps/99_run_all_paired_steps.py
```

本次共发现 14 个 paired run。所有 run 在目标轮次下均具备完整的 `R_low`、`L_low`、`R_high`、`L_high`、合成图、诊断报告和 per-image details；核心可比字段一致，无阻塞性产物缺口。

## 总体结论

纠偏排名第一的配置为：

```text
LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1er_0.5smv1_20260713-075255
label: cr=0.001 er=0.1 sm=0.5v1
```

该 run 是本批实验中唯一明显改善 R 绝对尺度的配置。其 `R_low -> I_high PSNR` 达到 `18.08 dB`，`R_high -> I_high PSNR` 达到 `30.07 dB`，`R/high mean ratio` 降至 `1.32`。相比其它 run，它显著降低了 `R` 过亮问题，但过亮比例仍为 `0.540`，说明分解仍未完全稳定。

不能使用 `R_low/R_high PSNR` 单独排名。本批 consistency-only 第一名为：

```text
LOLv2_1.0rh_0.3rl_0.05crh_0.05crl_0.3er_0.1smv3_20260717-160114
label: cr=0.05 er=0.3 sm=0.1v3
```

该 run 的 `R_low/R_high PSNR` 为 `26.07 dB`，但 `R_low -> I_high PSNR` 仅 `6.25 dB`，`R/high mean ratio` 为 `2.37`，过亮比例为 `0.978`。这正是 `R_low≈R_high≈过亮图` 的失败模式。

## 纠偏排名 Top 5

| Rank | 配置 | Rlow->high PSNR | Rhigh->high PSNR | R/high ratio | 过亮比例 | Rlow/Rhigh PSNR |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `cr=0.001 er=0.1 sm=0.5v1` | 18.08 | 30.07 | 1.32 | 0.540 | 19.62 |
| 2 | `cr=0.001 er=0.1 sm=0.3v1` | 9.24 | 9.92 | 1.99 | 0.960 | 22.24 |
| 3 | `cr=0.001 er=0.1 sm=0.3v2` | 9.03 | 9.69 | 2.02 | 0.959 | 22.73 |
| 4 | `cr=0.001 er=0.1 sm=0.1v2` | 7.74 | 8.18 | 2.16 | 0.971 | 24.17 |
| 5 | `cr=0.001 er=0.1 sm=0.1v1` | 7.68 | 8.15 | 2.17 | 0.973 | 23.97 |

## 参数趋势

- `sm=0.5v1` 在 `cr=0.001, er=0.1` 下表现最好，主要优势是 R 的绝对亮度尺度更接近 `I_high`。
- `sm=0.3v1/v2` 排名第二、第三，但 `R/high ratio` 仍接近 `2.0`，过亮比例接近 `0.96`。
- 大多数 `sm=0.1` 或更高 `cr/equal_R` 配置会提高 `R_low/R_high` 一致性，但同时让 R 明显过亮。
- `cr` 从 `0.001` 提高到 `0.01/0.05/0.1` 后，重建类指标仍较高，但分解质量变差，表现为 `R_low -> I_high PSNR` 下降、`R/high ratio` 升高。
- `equal_R` 权重增大可以推高 consistency-only 指标，但不能保证 R 的绝对尺度正确。

## L 分支现象

最佳 run 的 `L_high mean` 为 `0.919`，`corr(L_high, I)` 为 `0.463`，明显低于其它多数 run 的 `0.90+`。这说明该配置下 `L_high` 不再强烈贴合输入纹理，R 的绝对尺度也更合理。

其它大多数 run 的 `L_high mean` 约为 `0.44-0.52`，`corr(L_high, I)` 约为 `0.94-0.96`，同时 R 大面积过亮。这更接近一种不理想的恒等式分解：L 携带较多图像结构，R 通过过亮补偿重建。

## 训练动态

训练日志共解析到 280 条 eval 记录。14 个 run 全部出现如下现象：

```text
total loss 下降，但 R consistency proxy 大幅下降 25-34 dB
```

因此，不能只根据 total loss、self reconstruction 或 cross reconstruction 选择配置或 checkpoint。当前 paired 实验必须以纠偏后的 `R -> I_high` 绝对尺度指标为主，再辅以 `R_low/R_high` consistency 和视觉网格判断。

## 视觉检查

自动选出的视觉样本显示：

- 典型样本中，最佳 run 的 R 更接近 high 图；其它配置虽然 R_low/R_high 更一致，但 R 明显发白。
- 最坏 highref 样本中，最佳 run 仍存在暗平面场景过亮问题，`overbright=0.993`，但比其它配置更可控。

关键视觉图：

- `figures/typical_highref_index_85.png`
- `figures/worst_highref_index_73.png`
- `figures/worst_consistency_index_36.png`
- `figures/worst_overbright_index_73.png`

## 推荐判断

当前批次中可作为后续基线的配置是：

```text
cr=0.001 er=0.1 sm=0.5v1
```

但它仍不是完全满意的 Retinex 分解结果。下一步优化应优先围绕“降低 R 过亮、减少 L 结构泄漏、避免 loss 与分解质量脱钩”展开，而不是继续单纯提高 `R_low/R_high` consistency。

## 输出文件

主要结果文件：

- `inventory.md`
- `corrected_analysis.md`
- `corrected_ranking.csv`
- `corrected_summary.csv`
- `training_dynamics.md`
- `training_eval.csv`
- `selected_visual_cases.csv`
- `figures/*.png`
