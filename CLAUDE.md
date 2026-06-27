# CLAUDE.md

## 项目简介

Retinex-TDN：基于 Retinex 理论与小波 Transformer 的低光图像分解增强网络。DecomNet 将低光图像分解为反射分量 R 和光照分量 L，使用 DWT-FSA 注意力机制进行多尺度频域特征建模。

## 关键文件

| 文件 | 职责 |
|---|---|
| `train.py` | 训练入口，支持 YAML 配置 |
| `test.py` | 推理与跨分量重组评估 |
| `utils.py` | 训练辅助函数 |
| `models/network.py` | RetinexPoint + TDN backbone |
| `models/wavelets.py` | 可微 DWT / IDWT 小波变换 |
| `loss/decomposition_loss.py` | Retinex 分解总损失 |
| `loss/bsdp.py` | BDSP 边缘检测算子 |
| `data/dataset.py` | 配对/非配对/纯低光 Dataset |
| `configs/` | YAML 配置文件（按训练模式分子目录） |
| `scripts/metrics.py` | PSNR / SSIM / LPIPS 指标计算 |

## 网络类（model.name）

配置文件中 `model.name` 可选以下三个网络类，均定义在 `models/network.py`：

| 网络类 | L 分支输出 | 说明 |
|---|---|---|
| `RetinexPointRaw` | 标量 `L [B,1,1,1]` | 默认模型。R 分支为 TDN Transformer U-Net，L 分支从多尺度特征池化后经全连接输出全局光照标量 |
| `RetinexPixelClassic` | 逐像素 `L [B,1,H,W]` | R 分支同上，L 分支替换为轻量 CNN（3 层 Conv），与 Diff-TDN 原版一致 |
| `RetinexPixelTrans` | 逐像素 `L [B,1,H,W]` | R 分支同上，L 分支从 encoder 中间特征出发，经两次上采样 + DWTTransformer 处理后输出逐像素光照图 |

## 常用命令

```bash
# 训练
python train.py --config configs/paired/lol_exp1.yaml

# 推理（需先在 test.py 中修改权重路径与数据路径）
python test.py

# 指标计算
python -m scripts.metrics --pair <预测文件夹> <真值文件夹>

# TensorBoard
tensorboard --logdir experiments/
```

## 损失函数架构

### 损失模式（8种）

`loss.mode` 必须显式设置，格式为 `{data_mode}_{l_type}`：

| 数据模式 | L 类型 | 说明 |
|---|---|---|
| `paired` | `_point` / `_pixel` | 有监督成对训练 |
| `unpaired` | `_point` / `_pixel` | 非配对训练 |
| `pure_low_double` | `_point` / `_pixel` | 纯低光双视图自监督 |
| `pure_low_single` | `_point` / `_pixel` | 纯低光单视图 |

## 注意事项

- 数据目录必须严格遵循 `train/low`, `train/high`, `test/low`, `test/high` 结构
- 修改模型结构时同步检查 `models/wavelets.py` 中的 DWT/IDWT 兼容性
- `scripts/` 下工具通过 `python -m scripts.xxx` 调用

### 光度增强配置（pure_low_double 模式）

在 `data.photometric_augment` 段配置，仅训练时启用，对两个 view 独立应用：

```yaml
data:
  photometric_augment:
    enabled: true
    gamma_range: [0.7, 1.5]        # gamma 校正范围
    brightness_range: [0.6, 1.4]    # 亮度缩放范围
```

## 配置文件结构

```
configs/
├── LOLv2_base.yaml              # 顶层基础配置
├── paired/                      # 成对训练配置
├── unpaired/                    # 非配对训练配置
├── pure_low_double/             # 纯低光双视图配置
└── pure_low_single/             # 纯低光单视图配置
```
