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

## 代码规范

- 使用中文撰写 README 和文档注释
- 配置参数通过 YAML 管理，不要硬编码路径或超参数
- 训练输出统一写入 `experiments/`，推理结果写入 `results/`
- 模型输出 R ∈ [0,1]（sigmoid），L ∈ [0,1]（sigmoid 标量广播）
- 损失函数权重在配置文件中通过 `loss` 段控制，缩写规则见 README

## 注意事项

- 数据目录必须严格遵循 `train/low`, `train/high`, `test/low`, `test/high` 结构
- 修改模型结构时同步检查 `models/wavelets.py` 中的 DWT/IDWT 兼容性
- `scripts/` 下工具通过 `python -m scripts.xxx` 调用
