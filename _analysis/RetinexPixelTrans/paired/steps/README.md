# RetinexPixelTrans / paired 可复用分析步骤

这组脚本用于新的 `experiments/RetinexPixelTrans/paired/*` 实验完成后，直接复用当前已经确认过的 paired 分析流程，避免每轮重新探索指标。

前置习惯保持不变：实验结束后先运行：

```bash
bash _compare/run_all.sh
```

然后运行本目录的一键脚本：

```bash
.venv/bin/python _analysis/RetinexPixelTrans/paired/steps/99_run_all_paired_steps.py
```

默认分析 `iteration=10000`。如果要分析其它轮次：

```bash
.venv/bin/python _analysis/RetinexPixelTrans/paired/steps/99_run_all_paired_steps.py --iteration 5000
```

如果要重新强制生成明细：

```bash
.venv/bin/python _analysis/RetinexPixelTrans/paired/steps/99_run_all_paired_steps.py --force
```

输出位置：

```text
_analysis/RetinexPixelTrans/paired/steps/results/
```

## 步骤定义

### 00_inventory.py

盘点 `experiments/RetinexPixelTrans/paired` 下所有 run：

- 是否存在 `config.yaml`、`train.log`、`img/<iteration>`；
- `R_low/L_low/R_high/L_high` 是否齐全；
- paired 可比性字段：数据集、seed、crop、batch、loss 权重、smooth 版本；
- checkpoint、合成图、`decomposition_analysis` 基础产物是否存在。

输出：

- `inventory.csv`
- `inventory.md`

### 01_prepare_details.py

对每个 paired run 调用 `_compare/analyze_decomposition.py --details --iteration <N>`，确保每个实验目录下都有：

- `decomposition_analysis.txt`
- `decomposition_analysis_details.csv`

注意：`_compare/run_all.sh` 默认只生成 txt，不一定生成 per-image details，因此本步骤会补齐 details。

### 02_summarize_rank.py

读取每个 run 的 `decomposition_analysis_details.csv`，生成纠偏后的 paired 汇总和排名。

核心原则来自当前 `Summary.md` 的修正结论：

- 不能只看 `R_low` 与 `R_high` 是否一致；
- 必须同时看 `R_low -> I_high`、`R_high -> I_high` 的绝对尺度与颜色/亮度接近程度；
- `R_low≈R_high≈过亮图` 不能被误判为好分解。

优先报告：

- `r_low_highref_psnr_mean`
- `r_high_highref_psnr_mean`
- `r_low_highref_mean_ratio_mean`
- `r_low_highref_overbright_010_mean`
- `r_consistency_psnr_mean`
- `l_high_mean_mean`
- `l_low_input_gray_corr_mean`
- `l_high_input_gray_corr_mean`
- self/cross reconstruction 指标

输出：

- `corrected_summary.csv`
- `corrected_ranking.csv`
- `corrected_analysis.md`

### 03_training_dynamics.py

解析 `train.log` 中的 eval 行，检查训练过程是否出现：

- total loss 下降但 `R_low/R_high` proxy PSNR 同时崩塌；
- equal_R、smooth、cross_recon 与 R 一致性之间的相变；
- checkpoint 选择是否只偏向重建 loss。

输出：

- `training_eval.csv`
- `training_dynamics.md`
- `figures/training_total_proxy.png`

### 04_make_visual_grids.py

基于 `02_summarize_rank.py` 的纠偏排名和 per-image details 自动选择视觉样本：

- 当前最佳 run 的典型样本；
- 当前最佳 run 的最差 `R_low -> I_high` 样本；
- 当前最佳 run 的最差 `R_low/R_high` 一致性样本；
- 当前最佳 run 的最严重过亮样本。

并横向对比 top runs，展示：

- input low；
- input high；
- `R_low`；
- `R_high`；
- `L_low`；
- `L_high`；
- `|R_low - R_high|`；
- `|R_low - I_high|`。

输出：

- `selected_visual_cases.csv`
- `figures/*.png`

## 判断顺序

后续 paired 实验分析按以下顺序读结果：

1. 先看 `inventory.md`，确认实验产物完整且配置可比。
2. 再看 `corrected_analysis.md`，以 `R -> I_high` 绝对尺度指标作为主判据。
3. 再看 `R_low/R_high` consistency，判断曝光不变性是否同步成立。
4. 再看 L 结构泄漏：`corr(L,I)`、`TV(L)/TV(I)`、`L_high_mean`。
5. 最后看视觉图，确认是否存在发白、偏色、L 携带纹理或 `L_high≈1/R_high≈I_high` 的恒等分解。

## 关键避免项

不要再单独用 `R_low/R_high PSNR` 给 run 排名。该指标只能说明两个 R 是否一致，不能说明 R 的绝对尺度正确。当前项目里已经出现过 `R_low≈R_high≈过亮图` 的情况，这会让一致性分数升高，但真实分解质量下降。
