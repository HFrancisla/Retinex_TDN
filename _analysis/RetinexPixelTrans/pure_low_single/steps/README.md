# RetinexPixelTrans / pure-low-single 可复用分析步骤

这组脚本用于新的 `experiments/RetinexPixelTrans/pure_low_single/*` 实验完成后，复用当前已经确认过的 pure-low-single 分析流程。

推荐顺序仍是先跑通 `_compare` 的基础流水线：

```bash
bash _compare/run_all.sh
```

然后运行本目录的一键分析：

```bash
.venv/bin/python3 _analysis/RetinexPixelTrans/pure_low_single/steps/99_run_all_pure_low_single_steps.py
```

默认分析 `iteration=10000`。如果要分析其它轮次：

```bash
.venv/bin/python3 _analysis/RetinexPixelTrans/pure_low_single/steps/99_run_all_pure_low_single_steps.py --iteration 8000
```

如果 `_compare/analyze_decomposition.py` 的字段更新过，或旧 details 缺少 high-reference / anchor / input-structure 字段：

```bash
.venv/bin/python3 _analysis/RetinexPixelTrans/pure_low_single/steps/99_run_all_pure_low_single_steps.py --force
```

输出位置：

```text
_analysis/RetinexPixelTrans/pure_low_single/steps/results/
```

## pure-low-single 与 paired 的关键区别

pure-low-single 没有 paired 模式中的 `R_low/R_high` 曝光一致性对象，也没有 cross reconstruction。因此不能复用 paired 的排名逻辑。

本流程重点回答：

- `R×L` 是否能重建输入；
- R 是否塌缩、饱和或过亮；
- R 是否放大低光噪声、高频纹理和压缩伪影；
- L 是否复制输入灰度/物体结构；
- anchor 是否真正约束住 L 尺度；
- 改 recon / anchor / smooth / smooth_version 后 R 是否稳定；
- 对 LOLv2，R 是否比 low 更接近同序号 high。这里 high 只作为诊断参考，不是严格反射率真值；
- 对 BDDnight，只做 no-reference 和全量验证损失诊断，不做 high-reference 排名。

## 步骤定义

### 00_inventory.py

盘点 `experiments/RetinexPixelTrans/pure_low_single` 下所有 run：

- 配置字段：dataset、recon、anchor、BDSP、smooth、smooth_version；
- `img/<iteration>` 是否存在；
- `R_low/L_low` 是否齐全；
- synthesis、decomposition report、details、final full validation、checkpoint 是否存在；
- 训练中/失败 run 会保留在 inventory 中，但不参与后续排名。

输出：

- `inventory.csv`
- `inventory.md`

### 01_prepare_details.py

调用 `_compare/analyze_decomposition.py --details --iteration <N>`，补齐或刷新：

- `decomposition_analysis.txt`
- `decomposition_analysis_details.csv`

如果已有 details 缺少以下关键字段，会自动强制刷新：

- `self_low_psnr`
- `r_low_tv_to_input`
- `l_low_input_gray_corr`
- `anchor_abs_error`
- LOLv2 可选 high-reference 字段：`r_low_highref_psnr`

### 02_summarize_rank.py

按数据集分别汇总和排名。

LOLv2 使用 high-reference 诊断指标，但不把 high 当作反射率真值：

- `r_low_highref_psnr_mean`
- `r_low_highref_psnr_gain_vs_input_mean`
- `r_low_highref_mean_ratio_mean`
- `r_low_highref_overbright_010_mean`

所有数据集共同使用：

- `self_low_psnr_mean`
- `r_low_tv_to_input_mean`
- `l_low_input_gray_corr_mean`
- `l_low_tv_to_input_mean`
- `anchor_abs_error_mean`
- `r_low_bright_095_mean`
- `final_full_validation.yaml` 中的 full-resolution raw recon loss（如果存在）

输出：

- `pure_single_summary.csv`
- `pure_single_ranking.csv`
- `pure_single_analysis.md`

### 03_cross_run_stability.py

以每个数据集内的 `recon=1.0, anchor=v2, smooth=0` 作为基线，计算每个 run 相对基线的：

- R L1 / PSNR；
- L L1；
- R mean delta；
- R TV/input delta；
- L corr delta。

这一步用于判断“改损失权重后 R 是否仍稳定”。pure-low-single 中跨配置 R 大幅变化，通常说明分解仍欠约束。

输出：

- `cross_run_stability.csv`
- `cross_run_stability.md`

### 04_training_dynamics.py

解析 `train.log` eval 行：

- total/recon/anchor/bdsp/smooth weighted loss；
- 识别 total loss 降低但 raw recon 或分解指标不一定更好的情况；
- 合并 `final_full_validation.yaml` 的全量验证结果。

输出：

- `training_eval.csv`
- `final_full_validation_summary.csv`
- `training_dynamics.md`
- `figures/training_loss_curves.png`

### 05_make_visual_grids.py

按 `02` 的每数据集排名自动选择视觉样本：

- typical：当前数据集最佳 run 的中位样本；
- worst_recon：重建最差样本；
- worst_noise：`R TV / input TV` 最高样本；
- worst_l_leakage：`corr(L, gray(I))` 最高样本；
- LOLv2 额外选择 worst_highref：`R -> high` 最差样本。

每张拼图横向比较同数据集 top runs，列包括：

- input low；
- high ref（仅 LOLv2）；
- R；
- L；
- `R×L`；
- `|R×L-input|`；
- `|R-high|`（仅 LOLv2）。

输出：

- `selected_visual_cases.csv`
- `figures/*.png`

## 判断顺序

后续新实验建议按以下顺序读：

1. `inventory.md`：先排除训练中、缺 img、R/L 数量不完整、details 过旧的 run。
2. `pure_single_analysis.md`：按数据集分别看，不要把 LOLv2 和 BDDnight 混排。
3. `cross_run_stability.md`：确认新配置是否只是靠改变 R/L 尺度或把噪声推入 R 得到低 loss。
4. `training_dynamics.md`：检查 total weighted loss 是否被权重缩放误导，尤其 recon_weight=0.3 的 run。
5. 视觉图：确认 R 噪声、L 结构泄漏、过亮/偏色、BDDnight 雨夜伪影。

## 关键避免项

- 不要使用 paired 的 `R_low/R_high` 指标；pure-low-single 没有对应对象。
- 不要用 weighted total loss 跨 `recon_weight` 直接比较好坏；权重不同会改变数值尺度。
- 不要把 LOLv2 high 当作严格 R 真值；它只是同场景正常曝光诊断参考。
- 不要只看 `R×L` 重建 PSNR；乘积正确不能证明 R/L 语义正确。
