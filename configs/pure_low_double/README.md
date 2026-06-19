# Pure Low Double 配置

保留配置：

- `base.yaml`：默认双视图配置
- `v2.yaml`：在 base 基础上开启 `reflect_weight`

典型命令：

```bash
python train.py --config configs/pure_low_double/base.yaml
python train.py --config configs/pure_low_double/v2.yaml
```
