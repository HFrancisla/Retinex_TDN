# 配置文件使用说明

## 目录结构

```text
configs/
├── README.md
├── base.yaml
├── paired/
│   ├── base.yaml
│   └── lol_exp1.yaml
├── unpaired/
│   └── base.yaml
├── pure_low_single/
│   ├── README.md
│   └── base.yaml
└── pure_low_double/
    ├── README.md
    ├── base.yaml
    └── v2.yaml
```

## 模式说明

本项目当前支持四种训练模式：

- `paired`：low/high 配对数据
- `unpaired`：low/high 非配对数据
- `pure_low_single`：仅 low 图像，单次增强，单视图自监督分解
- `pure_low_double`：仅 low 图像，两次增强，双视图自监督分解

## 推荐用法

### paired

```bash
python train.py --config configs/paired/lol_exp1.yaml
```

### unpaired

```bash
python train.py --config configs/unpaired/base.yaml
```

### pure_low_single

```bash
python train.py --config configs/pure_low_single/base.yaml
```

### pure_low_double

```bash
python train.py --config configs/pure_low_double/base.yaml
python train.py --config configs/pure_low_double/v2.yaml
```

## 创建新实验

1. 从对应模式目录复制 `base.yaml`
2. 修改实验名称和参数
3. 运行训练：

```bash
cp configs/pure_low_double/base.yaml configs/pure_low_double/my_new_exp.yaml
python train.py --config configs/pure_low_double/my_new_exp.yaml
```

## 最佳实践

1. **明确模式**：优先使用 `pure_low_single` / `pure_low_double`
2. **先单后双**：先验证单视图是否可学，再切换双视图
3. **保持配置最小化**：只保留有实质差异的实验配置
4. **保留 auto_name**：实验目录自动区分模式
