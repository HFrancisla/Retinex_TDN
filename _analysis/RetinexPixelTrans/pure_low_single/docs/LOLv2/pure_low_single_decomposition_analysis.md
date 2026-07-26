# RetinexPixelTrans pure-low-single 分解效果分析 (Numerical & Visual)

## 1. 数值指标分析 (Numerical Analysis)

### 1.1 数据集 LOLv2 评估
对于 LOLv2，情况有所不同，主要关注点是 R 能否在没有任何 ground truth 的情况下接近正常光照图像（high-reference）。

- **最高得分基线 (Conservative Top)**: `LOLv2_1.0r_0.05anchorv2_0.1bdsp_0.1smv2` (简称 `r1-a2-sm0.1v2`)。此版本的 `R TV/input` 约为 10.09，而 `corr(L,I)` 保持在 0.883。最关键的是，它的 R -> high PSNR 达到了 15.19 dB，在所有运行中表现最好，说明 R 层在无监督情况下相对最接近真实亮光图像。
- **不同 anchor 版本的影响**: `v1` (如 `r1-a1-sm0v1`) 有很高的 Self PSNR (41.98 dB)，但 anchor v2 (目前默认) 在防止 R 层色彩漂移/过曝方面通常更有优势。
- **稳定性**: 在 `cross_run_stability.md` 中可以看到 LOLv2 上的模型非常稳定，ΔR mean 和 ΔR TV/input 在大部分 1.0r 变体中仅在小范围内波动。

## 2. 视觉效果分析 (Visual Analysis)

在 `_analysis/RetinexPixelTrans/pure_low_single/steps/results/LOLv2/figures` 中包含了几类典型和极端情况的可视化图。

### 2.1 典型分解表现 (Typical Cases)

![LOLv2 Typical 1](_analysis/RetinexPixelTrans/pure_low_single/steps/results/LOLv2/figures/lolv2_typical_index_22.png)
![LOLv2 Typical 2](_analysis/RetinexPixelTrans/pure_low_single/steps/results/LOLv2/figures/lolv2_typical_index_37.png)
**结论**: 在正常的典型样本中，R 层成功提亮了图像并呈现出了隐藏的色彩，而 L 层清晰地捕获了光照强度分布（高光区 L 亮，暗区 L 暗）。重构图像 S (S = R * L) 与原低光图像保持了高度的视觉一致性。

### 2.2 糟糕的光照泄漏 (Worst L Leakage)

![LOLv2 Worst L Leakage 1](_analysis/RetinexPixelTrans/pure_low_single/steps/results/LOLv2/figures/lolv2_worst_l_leakage_index_21.png)
**结论**: 当出现 L Leakage（L 泄漏）时，本应该由 R 层承担的物体纹理或反照率特征错误地进入了 L 层。这导致 R 层过于平滑或出现色块，而 L 层看起来就像是一张去除了颜色的灰度图，失去了纯粹“光照图”的物理意义。

### 2.3 R层噪声严重 (Worst Noise in R)

![LOLv2 Worst Noise](_analysis/RetinexPixelTrans/pure_low_single/steps/results/LOLv2/figures/lolv2_worst_noise_index_98.png)
**结论**: 这是当正则化约束不足或 `recon_weight` 过低时常发生的现象（对应前文数值分析中 `R TV/input` 极高的情况）。模型为了满足重构，强行将原图中的噪声和传感器噪点全部挤压到 R 层，导致反射图 R 充满了严重的彩色噪点。

### 2.4 重构失败 (Worst Reconstruction)

![LOLv2 Worst Recon](_analysis/RetinexPixelTrans/pure_low_single/steps/results/LOLv2/figures/lolv2_worst_recon_index_70.png)
**结论**: 虽然纯低光单视图方法着重分解，但重构 (R*L) 必须尽可能贴近 Input。在一些极端分布样本上，由于 R\_sat (防过曝) 等强制先验，导致 S 和 Input 在亮度和对比度上产生明显差异。

### 2.5 偏离参考高光 (Worst High-ref for LOLv2)

![LOLv2 Worst Highref](_analysis/RetinexPixelTrans/pure_low_single/steps/results/LOLv2/figures/lolv2_worst_highref_index_73.png)
**结论**: 由于 `pure_low_single` 是单视图无监督训练，没有任何成对高光参考。因此，在某些包含复杂光源的场景中，预测的 R 层虽然自恰，但在绝对颜色映射上与真实的 Ground Truth (High) 相差甚远（如颜色发灰、饱和度不足）。数值上对应了 `R->high PSNR` 较低的情况。

---
## 3. 总体结论

> [!TIP]
> 综合上述数值与视觉效果：
> 1. `pure_low_single` 分支已经能够较好地在**无监督**条件下分离光照 L 与反射 R，对于大部分正常数据，视觉分解符合 Retinex 物理含义。
> 2. **最敏感的超参是 `recon_weight (r)`**。降低重建权重会导致 R 层狂吃噪声（Noise），而维持在 `r=1.0` 则能提供最稳定的分解。
> 3. L Leakage 是目前的挑战之一，需要在后续进一步微调 `smooth (sm)` 或增强 `R_consistency`。
