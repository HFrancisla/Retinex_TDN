# RetinexPixelTrans pure-low-single 实验分析报告

分析日期：2026-07-19  
分析轮次：`iteration=10000`  
实验范围：`experiments/RetinexPixelTrans/pure_low_single`

## 结论摘要

本次共纳入 14 个已完成实验，其中 BDDnight 5 个、LOLv2 9 个。所有 run 均具备 `img/10000`、`R_low`、`L_low`、synthesis 和 decomposition details，默认分析路径无阻塞缺口。

当前推荐按数据集分别选择：

- BDDnight：`BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260715-233953`
- LOLv2：`LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2_20260717-200532`

核心判断是：`R x L` 重建低光输入整体可靠，但 pure-low-single 下的分解仍欠约束。R 普遍过亮，并会放大低光噪声、高频纹理和压缩伪影；L 中仍有明显输入灰度/物体结构泄漏。因此不能只依据重建 PSNR 判断分解质量。

## BDDnight 结果

BDDnight 不使用 high-reference 指标，排序主要依据自重建、R 噪声/纹理放大、L 结构泄漏、anchor 误差、R 饱和比例和 full-resolution recon。

| 排名 | run | self PSNR | R TV/input | corr(L,I) | anchor err | R>0.95 | full recon |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260715-233953` | 41.41 | 3.32 | 0.983 | 0.0440 | 0.035 | 0.00355 |
| 2 | `BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-060543` | 40.95 | 3.53 | 0.981 | 0.0464 | 0.026 | 0.00358 |
| 3 | `BDDnight_1.0r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-081309` | 39.83 | 3.91 | 0.973 | 0.0488 | 0.021 | 0.00378 |
| 4 | `BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-102043` | 36.88 | 4.99 | 0.979 | 0.0298 | 0.104 | 0.00624 |
| 5 | `BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-123043` | 34.00 | 22.23 | 0.977 | 0.0258 | 0.122 | 0.00900 |

BDDnight 的最佳配置是 `recon=1.0, anchor=v2, bdsp=0.05, smooth=0`。加入 smooth 后 full recon 和 self PSNR 略降；将 `recon_weight` 降到 0.3 后，重建和稳定性明显变差。尤其 `r0.3-a2-sm0.5v1` 的 `R TV/input=22.23`，说明 R 中噪声和局部纹理被显著放大。

跨 run 稳定性也支持这一结论：相对 BDDnight 基线，`r1-a2-sm0.1v1` 的 `R L1=0.0116`，`r1-a2-sm0.5v1` 的 `R L1=0.0272`，而两个 `recon_weight=0.3` run 分别升至 `0.0715` 和 `0.1092`。

## LOLv2 结果

LOLv2 使用 high-reference 作为诊断参考，但 high 不作为严格反射率真值。排序仍按 pure-low-single 口径，避免与 paired 实验混排。

| 排名 | run | self PSNR | R TV/input | corr(L,I) | anchor err | R->high PSNR | R/high |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv2_20260717-200532` | 41.67 | 10.96 | 0.889 | 0.0355 | 13.84 | 1.42 |
| 2 | `LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1_20260714-024244` | 41.98 | 10.97 | 0.895 | 0.0229 | 13.86 | 1.41 |
| 3 | `LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv3_20260718-094046` | 41.67 | 10.98 | 0.891 | 0.0353 | 13.80 | 1.42 |
| 4 | `LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1_20260714-060043` | 41.93 | 11.00 | 0.895 | 0.0356 | 13.83 | 1.42 |
| 5 | `LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1_20260714-155441` | 41.61 | 11.00 | 0.891 | 0.0350 | 13.76 | 1.43 |
| 6 | `LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.5smv1_20260714-191342` | 41.19 | 11.20 | 0.879 | 0.0341 | 13.58 | 1.44 |
| 7 | `LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260714-223238` | 37.00 | 11.53 | 0.893 | 0.0176 | 12.03 | 1.64 |
| 8 | `LOLv2_1.0r_0.05anchorv2_0.0bdsp_0.1smv3_20260718-130429` | 36.88 | 2.61 | 0.925 | 0.0090 | 7.42 | 2.14 |
| 9 | `LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260715-015134` | 36.60 | 11.70 | 0.874 | 0.0181 | 12.06 | 1.64 |

LOLv2 推荐 `smooth_version=v2, smooth_weight=0.1` 的 v2 anchor 配置。`smv3` 与 `smv2` 接近，但 `R->high PSNR` 略低；`smooth=0` 的自重建略高，但综合分解诊断不占优。

`anchor=v1` 的 run 指标接近甚至部分更高，但其 anchor 目标与当前 v2 不同，排序中已施加 canonical-anchor penalty。它适合作为历史 ablation 参考，不建议作为当前主配置。

去掉 BDSP 的 run 是明确负例：`R L1=0.2591`、`R PSNR=10.47`、`R->high PSNR=7.42`、`R/high=2.14`。这表明 BDSP 对 pure-low-single 的 R/L 尺度约束很关键；没有 BDSP 时，模型会得到更小 anchor error，但反射分量语义明显失真。

## 训练动态与比较口径

本次解析到 280 条 eval 记录，BDDnight 有 5 个 full-validation summary。跨 `recon_weight` 比较时必须看未加权的 `full_recon_loss`，不能直接比较 weighted total loss。

两个 BDDnight `recon_weight=0.3` run 的 weighted recon 看似较小，但 full-resolution recon 明显更差：

- `BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.1smv1_20260716-102043`：full recon `0.00624`，weighted recon `0.00187`
- `BDDnight_0.3r_0.05anchorv2_0.05bdsp_0.5smv1_20260716-123043`：full recon `0.00900`，weighted recon `0.00270`

因此后续比较不同 `recon_weight` 的实验时，应优先使用 unweighted raw/full recon 指标，并结合 R 稳定性和视觉诊断。

## 视觉诊断

视觉拼图显示：

- `R x L` 基本能够重建 low，重建误差主要集中在边缘、高亮点和局部纹理；
- LOLv2 的 R 往往比 low 和 high reference 更亮，`R/high` 约 1.4，worst high-reference case 中差异更明显；
- BDDnight 的 R 会显著提亮道路、灯光和雨夜噪声，低 recon 或高 smooth 配置下噪声放大更严重；
- L 不是纯光照图，仍存在较强场景结构泄漏，尤其 `corr(L,I)` 在 BDDnight 接近 0.98，在 LOLv2 约 0.89。

重点查看图：

- `figures/lolv2_typical_index_22.png`
- `figures/lolv2_worst_highref_index_73.png`
- `figures/bddnight_typical_index_221.png`
- `figures/bddnight_worst_noise_index_200.png`

## 后续建议

1. BDDnight 后续以 `r1-a2-sm0v1` 为当前基线，不建议继续降低 `recon_weight` 到 0.3。
2. LOLv2 后续以 `r1-a2-sm0.1v2` 为当前主配置，`smv3` 可作为近邻对照继续观察。
3. BDSP 不应移除；无 BDSP 会显著破坏 R/L 尺度和 R 语义。
4. 不要用 weighted total loss 跨权重比较实验好坏，应优先看 unweighted recon、R TV/input、`corr(L,I)`、R 饱和比例和视觉拼图。
5. pure-low-single 目前仍需要额外约束来改善分解语义，尤其是抑制 R 噪声放大和 L 结构泄漏。

## 产物索引

- `inventory.md`：实验完整性盘点
- `pure_single_analysis.md`：按数据集排名和主指标汇总
- `cross_run_stability.md`：相对基线的 R/L 稳定性
- `training_dynamics.md`：训练 eval 和 full-validation 诊断
- `selected_visual_cases.csv`：视觉样本选择
- `figures/`：典型和失败案例拼图
