# RetinexLR

基于 Retinex 理论与小波 Transformer 的低光图像分解增强方法。将低光图像分解为反射分量 R 和光照分量 L，通过 DWT-FSA 注意力机制实现多尺度频域特征建模。

## 项目结构

```
RetinexLR/
├── models/                        # 网络模型
│   ├── network.py                 # RetinexPoint + TDN backbone
│   └── wavelets.py                # 可微 DWT / IDWT 小波变换
├── data/                          # 数据加载
│   ├── dataset.py                 # 配对/非配对/纯低光 Dataset
│   └── transforms.py              # 同步数据增强
├── loss/                          # 损失函数
│   ├── decomposition_loss.py      # Retinex 分解总损失
│   └── bsdp.py                    # BDSP 边缘检测算子
├── configs/                       # YAML 配置文件（推荐用法）
├── scripts/                       # 辅助分析工具
│   ├── eval.py                    # 对数域梯度图评估
│   ├── metrics.py                 # PSNR / SSIM / LPIPS 指标计算
│   └── analyze_model.py           # 参数量与 FLOPs 分析
├── train.py                       # 训练入口
├── test.py                        # 推理与跨分量重组评估
├── utils.py                       # 训练辅助函数
├── experiments/                   # 训练输出（权重、日志）
└── results/                       # 推理结果输出
```

## 环境配置

需要 Python >= 3.8。

```bash
pip install -r requirements.txt

# 若平台需要特定 CUDA 构建，请按 PyTorch 官方指引覆盖安装
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 数据准备

数据集目录结构：

```
<dataset_root>/
├── train/
│   ├── low/       # 低光训练图像
│   └── high/      # 正常光训练图像（GT）
└── test/
    ├── low/       # 低光测试图像
    └── high/      # 正常光测试图像（GT）
```

图像格式支持 `.jpg` / `.png`。

> **注意：** 训练前请务必确认数据集目录结构与上述一致。代码按固定路径 `train/low`、`train/high`、`test/low`、`test/high` 读取数据，目录名或层级不对会直接报错。

## 训练

推荐使用 YAML 配置文件启动训练：

```bash
python train.py --config configs/paired/lol_exp1.yaml
```

详细配置说明见 [configs/README.md](configs/README.md)。

### 训练模式

通过 `data.mode` 和 `loss.mode` 控制：

| 模式 | 说明 |
|---|---|
| `paired` | low/high 配对数据，一一对应 |
| `unpaired` | low/high 非配对，随机匹配 |
| `pure_low_single` | 仅 low 图像，单视图自监督分解 |
| `pure_low_double` | 仅 low 图像，双视图自监督分解 |

### 实验命名

配置文件中 `experiment` 段控制实验目录名：

```yaml
experiment:
  name: "lol_baseline"   # 手动命名（auto_name=false 时必填）
  auto_name: true        # 自动生成
  tag: ""                # 可选后缀
```

**auto_name=true** 时自动生成格式：`{dataset}_{mode}_{损失权重缩写}`

- `dataset`：取 `data.path` 最后一段目录名
- `mode`：训练模式
- 非零损失权重按固定顺序拼接，格式为 `{值}{缩写}`

示例输出：

```
LOLv2_paired_1r_0.05anchor_0.05bdsp_0.05sr
```

完整目录名还会追加时间戳：`experiments/LOLv2_paired_1r_0.05anchor_0.05bdsp_0.05sr_20260619-153000/`

### 损失权重缩写

| 配置键 | 缩写 | 说明 |
|---|---|---|
| `recon_weight` | `r` | 重建损失 |
| `recon_weight_high` | `rh` | 高光重建 |
| `recon_weight_low` | `rl` | 低光重建 |
| `cross_recon_weight_high` | `crh` | 交叉重建（高光） |
| `cross_recon_weight_low` | `crl` | 交叉重建（低光） |
| `equal_r_weight` | `er` | 反射一致性 |
| `anchor_weight` | `anchor` | 光照锚定 |
| `bdsp_weight` | `bdsp` | BDSP 结构保持 |
| `smooth_weight` | `sm` | 光照平滑 |
| `self_recon_weight` | `sr` | 自重构约束 |
| `reflect_weight` | `ref` | 反射约束 |

### TensorBoard

```bash
tensorboard --logdir experiments/
```

## 推理与评估

```bash
# 推理 + 跨分量重组（需在 test.py 中修改权重路径与数据路径）
python test.py

# 图像质量指标计算
python -m scripts.metrics \
    --pair <预测图文件夹> <真值图文件夹>

# 多组批量对比
python -m scripts.metrics \
    --pair ./results/LLxLR ./datasets/low \
    --pair ./results/HLxHR ./datasets/high

# 模型参数量与 FLOPs 分析
python -m scripts.analyze_model

# 对数域梯度图评估
python -m scripts.eval
```

## 网络架构

**RetinexPoint** 接收低光图像，输出：

- **R**：反射分量（Reflectance），经 sigmoid 约束至 [0, 1]
- **L**：光照标量（Illumination），经 sigmoid 约束至 [0, 1]，广播至与输入同尺寸

核心组件：

- **TDN backbone**：U-Net 结构的 Transformer 编码器-解码器，3 级多尺度特征提取
- **DWT-FSA Attention**：基于离散小波变换的频域自注意力，LL 子带做 C×C 多头注意力，高频子带做 sigmoid 门控
- **DWT-FFN**：小波域前馈网络，仅处理 LL 子带，高频旁路直通

损失函数包含重建损失（R×L ≈ I）、光照锚定、BDSP 结构保持和光照平滑约束。
