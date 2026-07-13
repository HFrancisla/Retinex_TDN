## 项目简介

Retinex-TDN：基于 Retinex 理论与小波 Transformer 的低光图像分解网络。TDN（Transformer Decomposition Network）将低光图像分解为反射分量 R 和光照分量 L，利用 DWT-FSA（频域自注意力）进行多尺度小波域特征建模。`重建图 = R ⊙ L` 重组得到原始低光图像的重建结果（网络仅做分解，不直接输出增强图像）。

## 网络类（`model.name`）

4 个网络类均定义在 `models/network.py`，R 分支共用同一个 `TDN` backbone（3 级 U-Net + DWT-FSA Transformer），区别仅在于 L 分支：

| 网络类 | L 输出 | L 分支结构 |
|---|---|---|
| `RetinexPointRaw` | 标量 `[B,1,1,1]` | 多尺度特征 cat → 2 层 depthwise Conv + BN → AdaptiveAvgPool → FC → sigmoid |
| `RetinexPixelClassic` | 逐像素 `[B,1,H,W]` | conv0(3→C) → 3×Conv+ReLU → recon(C→1) → sigmoid |
| `RetinexPixelTrans` | 逐像素 `[B,1,H,W]` | cat(fea_L3, fea_down1) → 1×1 reduce(8C→4C) → 2×Upsample → DWTTransformer → Conv → sigmoid |
| `RetinexPixelTransMinus` | 逐像素 `[B,1,H,W]` | fea_down1 − fea_L3（逐像素做差）→ 2×Upsample → DWTTransformer → Conv → sigmoid |

> `RetinexPixelTransMinus` 相比 `RetinexPixelTrans` 去掉了 1×1 降维卷积，改为特征做差融合。

## 损失函数

`loss.mode` 格式为 `{数据模式}_{L类型}`，共 8 种组合：

| mode | 损失类 | 核心损失项 |
|---|---|---|
| `paired_point` | `PairedLoss` | 自重建 + 交叉重建 + equal_R |
| `paired_pixel` | `PairedLoss` | 同上 + Retinex smooth |
| `unpaired_point` | `UnpairedLoss` | 重建 + anchor(全局) + BDSP + `redecomp_l_consistency` |
| `unpaired_pixel` | `UnpairedLoss` | 同上 + Retinex smooth |
| `pure_low_double_point` | `PureLowDoubleLoss` | 重建 + anchor + BDSP + `redecomp_l_consistency` + **reflect**(R1↔R2 一致性) |
| `pure_low_double_pixel` | `PureLowDoubleLoss` | 同上 + Retinex smooth |
| `pure_low_single_point` | `PureLowSingleLoss` | 重建 + anchor + BDSP |
| `pure_low_single_pixel` | `PureLowSingleLoss` | 同上 + Retinex smooth |

**关键区别：**

- **`_point`**：L 为标量，anchor 约束 `L ≈ max(I)`，无 smooth
- **`_pixel`**：L 为逐像素图，anchor 仅约束 `mean(L) ≈ mean(I)`，带 Retinex smooth
- **`pure_low_double`** 独有 `reflect_weight`：约束两个增强 view 的 R 分量一致性（带 detach 防梯度泄露）

## 数据集类

| 类 | 数据要求 | 说明 |
|---|---|---|
| `MyDataSet` | `train/low` + `train/high` 配对 | 同步空间增强 |
| `UnpairedDataSet` | `train/low` + `train/high`（不配对） | 各自独立增强 |
| `PureLowDataSet` | 仅 `train/low` | 双视图自监督，可选光度增强 |
| `PureLowSingleDataSet` | 仅 `train/low` | 单视图，仅 Retinex 分解约束 |

## 分解结果分析

使用 `_compare/analyze_decomposition.py` 分析指定实验保存的验证集 R/L：

```bash
.venv/bin/python _compare/analyze_decomposition.py <实验目录> --iteration 10000
```

结果默认保存到实验目录下的 `decomposition_analysis.txt`。paired 验证会分析 `R_low/L_low/R_high/L_high` 并计算 R_low/R_high 一致性
