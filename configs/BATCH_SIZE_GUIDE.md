# Batch Size 设置规则

## 核心原则

`有效图数 = batch_size × 每步 forward 次数`。不同模式下 forward 次数不同，需调整 batch_size 使有效图数一致，避免显存超限。

## Forward 次数

| 模式 | forward/步 | 说明 |
|---|---|---|
| `paired` | 2 | `model(I_low)` + `model(I_high)` |
| `unpaired` | 2 | 同上 |
| `pure_low_double` | 2 | `model(view1)` + `model(view2)` |
| `pure_low_single` | 1 | 仅 `model(I_low)` |

## 取值规则

| 数据集 | crop_size | 1-forward 模式 | 2-forward 模式 |
|---|---|---|---|
| LOLv2 | 384 | `batch_size: 8` | `batch_size: 4` |
| BDD100k | 512 | `batch_size: 4` | `batch_size: 2` |

> 等效图数：LOLv2 ~8，BDD100k ~4。crop 越大显存越高，所以 BDD100k 整体减半。

## 检查清单

- [ ] `paired` / `unpaired` / `pure_low_double` → batch_size 为同数据集 `pure_low_single` 的一半
- [ ] 修改 batch_size 时同步确认 `num_workers` 不超过 `batch_size`
