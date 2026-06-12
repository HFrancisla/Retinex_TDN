# RetinexLR

基于 Retinex 理论与小波 Transformer 的低光图像分解增强方法。将低光图像分解为反射分量 R 和光照分量 L，通过 DWT-FSA 注意力机制实现多尺度频域特征建模。

## 项目结构

```
RetinexLR/
├── models/                        # 网络模型
│   ├── network.py                 # DecomNet + TDN backbone
│   └── wavelets.py                # 可微 DWT / IDWT 小波变换
├── data/                          # 数据加载
│   ├── dataset.py                 # 配对图像 Dataset
│   └── transforms.py              # 同步数据增强
├── loss/                        # 损失函数
│   ├── decomposition_loss.py      # Retinex 分解总损失
│   └── bsdp.py                    # BDSP 边缘检测算子
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

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境和依赖。

```bash
# 安装 uv（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
uv pip install -r requirements.txt

# 安装 PyTorch（根据 CUDA 版本选择，示例为 CUDA 12.1）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

需要 Python >= 3.8。

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

图像格式支持 `.jpg` / `.png`，low 与 high 文件夹中的图像需一一对应。

## 训练

```bash
python train.py \
    --data-path <dataset_root> \
    --epochs 300 \
    --batch-size 2 \
    --lr 0.0001 \
    --gpu_id 0
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-path` | - | 数据集根目录 |
| `--epochs` | 300 | 训练总轮数 |
| `--batch-size` | 2 | 批次大小 |
| `--lr` | 0.0001 | 学习率 |
| `--weights` | `''` | 预训练权重路径 |
| `--resume` | `''` | 恢复训练的 checkpoint 路径 |
| `--use_dp` | False | 是否使用 DataParallel 多卡训练 |
| `--gpu_id` | `'2'` | 可见 GPU 编号 |

训练输出保存至 `experiments/<实验名>/`，包含 `weights/`、`img/`（中间可视化）、`log/`（TensorBoard 日志）。

启动 TensorBoard：

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

**DecomNet** 接收低光图像，输出：
- **R**：反射分量（Reflectance），经 sigmoid 约束至 [0, 1]
- **L**：光照标量（Illumination），经 sigmoid 约束至 [0, 1]，广播至与输入同尺寸

核心组件：
- **TDN backbone**：U-Net 结构的 Transformer 编码器-解码器，3 级多尺度特征提取
- **DWT-FSA Attention**：基于离散小波变换的频域自注意力，LL 子带做 C×C 多头注意力，高频子带做 sigmoid 门控
- **DWT-FFN**：小波域前馈网络，仅处理 LL 子带，高频旁路直通

损失函数包含重建损失（R×L ≈ I）、交叉光照一致性、BDSP 结构保持和光照平滑约束。
