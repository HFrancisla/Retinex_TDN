# 配置文件使用说明

## 目录结构

```
configs/
├── README.md          # 本文件
├── base.yaml          # 基础配置模板（所有参数的默认值）
├── lol_exp1.yaml      # 示例实验1：基线配置
└── lol_exp2.yaml      # 示例实验2：消融实验
```

## 使用方法

### 1. 使用配置文件训练（推荐）

```bash
# 使用示例配置
python train.py --config configs/lol_exp1.yaml

# 使用自定义配置
python train.py --config configs/my_experiment.yaml
```

### 2. 使用命令行参数（向后兼容）

```bash
python train.py --data-path /path/to/dataset --epochs 300 --batch-size 2 --lr 0.0001
```

## 配置文件格式

配置文件使用 YAML 格式，包含以下部分：

### experiment - 实验标识

```yaml
experiment:
  name: "lol_baseline"      # 实验名称
  auto_name: true           # 是否自动生成名称
  tag: ""                   # 可选标签
```

**auto_name 说明：**

- `false`: 使用 `name` 字段作为目录名
- `true`: 自动生成基于损失权重的名称，格式：`{name}_{recon}r_{anchor}an_{bdsp}bdsp_{sr}sr[_{tag}]`

**示例：**

- `auto_name: false`, `name: "my_exp"` → 目录名：`my_exp_20250101-120000`
- `auto_name: true`, `name: "lol"`, `loss.mode` 为 `paired` 时 → 目录名：`lol_paired_20250101-120000`；`unpaired` 时 → `lol_unpaired_20250101-120000`

### data - 数据配置

```yaml
data:
  path: "datasets/LOLv2"  # 数据集路径
  mode: "paired"            # 训练模式: "paired"（配对）或 "unpaired"（非配对）
  crop_size: 256                    # 随机裁剪大小
  batch_size: 2                     # 批次大小
  num_workers: 0                    # 数据加载线程数
```

### model - 模型配置

```yaml
model:
  name: "DecomNet"          # 模型名称
  use_dp: false             # 是否使用 DataParallel
  gpu_id: "0"               # GPU 编号
```

### training - 训练配置

```yaml
training:
  epochs: 300               # 训练轮数
  lr: 0.0001                # 学习率
  warmup: true              # 是否使用 warmup
  save_interval: 20         # checkpoint 保存间隔（epoch）
```

### loss - 损失权重

```yaml
loss:
  mode: "paired"            # 可选："paired" / "unpaired"；不写则跟随 data.mode

  # paired 模式参数（参考 Diff-TDN 风格）
  recon_weight_high: 1.0
  recon_weight_low: 0.3
  cross_recon_weight_low: 0.001
  cross_recon_weight_high: 0.001
  smooth_weight: 0.1
  equal_r_weight: 0.1

  # unpaired 模式参数（当前项目默认风格）
  # recon_weight: 1
  # anchor_weight: 0.05
  # bdsp_weight: 0.05
  # smooth_weight: 0
  # self_recon_weight: 0.05
```

### resume - 恢复训练

```yaml
resume:
  checkpoint: ""            # 恢复的 checkpoint 路径
  weights: ""               # 预训练权重路径
```

## 实验目录结构

训练完成后，实验目录结构如下：

```
experiments/
└── lol_20r_1an_1bdsp_1sr_20250101-120000/
    ├── config.yaml          # 保存的配置文件（便于复现）
    ├── img/                 # 可视化结果
    │   ├── 0/               # epoch 0 的结果
    │   ├── 20/              # epoch 20 的结果
    │   └── ...
    ├── weights/             # 模型权重
    │   ├── best_model.pth   # 最佳模型（基于验证 loss）
    │   ├── checkpoint_epoch0.pth
    │   ├── checkpoint_epoch20.pth
    │   └── ...
    └── log/                 # TensorBoard 日志
```

## 创建新实验

1. 复制 `base.yaml` 或现有配置文件
2. 修改实验名称和参数
3. 运行训练：

```bash
cp configs/lol_exp1.yaml configs/my_new_exp.yaml
# 编辑 my_new_exp.yaml
python train.py --config configs/my_new_exp.yaml
```

## 最佳实践

1. **使用 auto_name**: 设置 `auto_name: true` 时，目录名会自动带 `paired/unpaired`（来自 `loss.mode`，默认跟 `data.mode`），便于快速区分实验
2. **添加 tag**: 使用 `tag` 字段标记特殊实验，如 `"v2"`, `"ablation_recon10"`, `"debug"`
3. **保存配置**: 训练时会自动将配置保存到实验目录，便于复现
4. **版本控制**: 建议将配置文件纳入版本控制，便于追踪实验历史
