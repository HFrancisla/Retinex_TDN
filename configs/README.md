# 配置文件使用说明

## 目录结构

配置按 **模型 → 训练模式** 两级组织：

```text
configs/
├── README.md
├── RetinexPointRaw/           # 全局光照标量 L [B,1,1,1]
│   ├── paired/
│   │   └── LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq.yaml
│   └── pure_low_single/
│       ├── BDD_1.0r_0.05anchorv1_0.05bdsp.yaml
│       ├── BDD_1.0r_0.05anchorv2_0.05bdsp.yaml
│       ├── LOLv2_1.0r_0.05anchorv1_0.05bdsp.yaml
│       └── LOLv2_1.0r_0.05anchorv2_0.05bdsp.yaml
├── RetinexPixelClassic/       # 逐像素 L [B,1,H,W]（轻量 CNN）
│   ├── paired/
│   │   └── LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml
│   └── pure_low_single/
│       ├── BDD_1.0r_0.05anchorv1_0.05bdsp_0.0smv1.yaml
│       ├── BDD_1.0r_0.05anchorv2_0.05bdsp_0.0smv1.yaml
│       ├── LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1.yaml
│       └── LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1.yaml
├── RetinexPixelTrans/         # 逐像素 L [B,1,H,W]（Transformer）
│   ├── paired/
│   │   ├── LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml
│   │   ├── LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.3smv1.yaml
│   │   └── LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.5smv1.yaml
│   └── pure_low_single/
│       ├── BDD_0.3r_0.05anchorv2_0.05bdsp_0.1smv1.yaml
│       ├── BDD_0.3r_0.05anchorv2_0.05bdsp_0.5smv1.yaml
│       ├── BDD_1.0r_0.05anchorv1_0.05bdsp_0.0smv1.yaml
│       ├── BDD_1.0r_0.05anchorv2_0.05bdsp_0.0smv1.yaml
│       ├── BDD_1.0r_0.05anchorv2_0.05bdsp_0.1smv1.yaml
│       ├── BDD_1.0r_0.05anchorv2_0.05bdsp_0.5smv1.yaml
│       ├── LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.1smv1.yaml
│       ├── LOLv2_0.3r_0.05anchorv2_0.05bdsp_0.5smv1.yaml
│       ├── LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1.yaml
│       ├── LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1.yaml
│       ├── LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1.yaml
│       └── LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.5smv1.yaml
└── RetinexPixelTransMinus/    # 逐像素 L [B,1,H,W]（Transformer 变体）
    ├── paired/
    │   └── LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml
    └── pure_low_single/
        ├── BDD_1.0r_0.05anchorv1_0.05bdsp_0.0smv1.yaml
        ├── BDD_1.0r_0.05anchorv2_0.05bdsp_0.0smv1.yaml
        ├── LOLv2_1.0r_0.05anchorv1_0.05bdsp_0.0smv1.yaml
        └── LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.0smv1.yaml
```

## 模型说明

| 模型 | L 分支输出 | 说明 |
|---|---|---|
| `RetinexPointRaw` | 标量 `L [B,1,1,1]` | 默认模型，L 从多尺度特征池化后经 FC 输出全局光照标量 |
| `RetinexPixelClassic` | 逐像素 `L [B,1,H,W]` | L 分支为轻量 CNN（3 层 Conv），与 Diff-TDN 原版一致 |
| `RetinexPixelTrans` | 逐像素 `L [B,1,H,W]` | L 分支从 encoder 中间特征经上采样 + DWTTransformer 处理 |
| `RetinexPixelTransMinus` | 逐像素 `L [B,1,H,W]` | RetinexPixelTrans 的简化变体 |

## 模式说明

本项目支持四种训练模式：

| 模式 | 目录 | 数据要求 | 说明 |
|---|---|---|---|
| `paired` | `paired/` | low + high 配对 | 有监督成对训练 |
| `pure_low_single` | `pure_low_single/` | 仅 low 图像 | 单视图自监督分解 |

## 参数约定

- **batch_size**：LOLv2 数据集统一 `8`，BDD100k 数据集统一 `4`
- **crop_size**：LOLv2 为 `384`，BDD100k 为 `512`
- **training 超参**：所有配置统一 `max_iterations=10000`、`lr=1e-4`、`warmup=1000`
- **验证频率**：每 `500` step 验证一次，验证 batch size 固定为 `4`

### Smooth 版本

所有 Pixel loss（包括 `smooth_weight: 0.0`）必须显式配置：

```yaml
loss:
  smooth_weight: 0.1
  smooth_version: "v1"  # v1=Raw, v2=Current, v3=Compromise
```

| 版本 | 梯度与边界 | R 梯度平均 | R 是否 detach |
|---|---|---|---|
| `v1` | 2×2 Conv + 零填充 | 3×3 AvgPool | 否 |
| `v2` | 真实相邻差分 | 无 | 是 |
| `v3` | 真实相邻差分 | 3×3 replicate 平均 | 是 |

版本号紧跟 smooth 权重写入配置文件名和自动实验名，例如 `0.1smv1`。复现基线
使用 `v1`，后续机制消融可使用 `v2`/`v3`；Point loss 不接受
`smooth_version`。

### Anchor 版本

所有包含 anchor 的 loss 必须显式配置：

```yaml
loss:
  anchor_version: "v2"  # 或 "v1"
```

| L 类型 | `v1` | `v2` |
|---|---|---|
| Point | 标量 L 拟合逐像素 max-RGB map | 标量 L 拟合每张图的全局最大值 |
| Pixel | `mean(L)` 拟合 `mean(max-RGB)` | `mean(L)` 拟合整张 RGB 图像均值 |

`anchor_version` 会紧跟 anchor 写入自动实验名，例如 `0.05anchorv1` / `0.05anchorv2`。
所有包含 anchor 的配置文件名也必须使用相同格式；不允许保留无版本号的 `0.05anchor_...`。

### Anchor 基准对比配置

四个网络均为 LOLv2、BDD 各提供一对严格匹配的基准配置：

```text
*anchorv1_*.yaml  # anchor_version: v1
*anchorv2_*.yaml  # anchor_version: v2
```

同一对配置除 `loss.anchor_version` 外完全一致，可直接用于公平对比。

## 推荐用法

```bash
# paired
python train.py --config configs/RetinexPointRaw/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq.yaml
python train.py --config configs/RetinexPixelClassic/paired/LOLv2_1.0rh_0.3rl_0.001crh_0.001crl_0.1eq_0.1smv1.yaml

# pure_low_single
python train.py --config configs/RetinexPointRaw/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp.yaml
python train.py --config configs/RetinexPixelTrans/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp_0.1smv1.yaml
```

## 创建新实验

1. 选择模型目录下对应训练模式的子目录
2. 复制已有配置并修改实验名称和参数：

```bash
cp configs/RetinexPointRaw/pure_low_single/LOLv2_1.0r_0.05anchorv2_0.05bdsp.yaml \
   configs/RetinexPointRaw/pure_low_single/my_exp.yaml
python train.py --config configs/RetinexPointRaw/pure_low_single/my_exp.yaml
```

## 最佳实践

1. **选对模型**：先确定 L 分支类型（Point / Pixel），再选对应模型目录
2. **先单后双**：先验证 pure_low_single 是否可学，再切换 pure_low_double
3. **保持配置最小化**：只保留有实质差异的实验配置
4. **保留 auto_name**：实验目录自动区分模式与模型
