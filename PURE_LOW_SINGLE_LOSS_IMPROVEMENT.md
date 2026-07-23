# RetinexPixelTrans / pure_low_single 损失函数改进方案

本文档整理针对 `RetinexPixelTrans/pure_low_single` 光照分离不佳问题的损失函数分析与修改方案。目标不是让 `R x L` 更容易重建低光输入，而是让 `R` 和 `L` 的语义更接近 Retinex 分解：`R` 保留相对稳定的反射/结构，`L` 表达低频光照分布，噪声和局部纹理不要大量泄漏到错误分支。

## 1. 当前问题

当前 `PureLowSingleLoss` 位于 `loss/decomposition_loss.py`，核心形式为：

```text
L_total =
  recon_weight * |R * L - I|
+ anchor_weight * anchor(L, I)
+ bdsp_weight * |BDSP(R) - BDSP(I)|
+ smooth_weight * smooth(L, R)
```

对于 `RetinexPixelTrans/pure_low_single`，模型输出：

```text
R: [B, 3, H, W], sigmoid 到 [0, 1]
L: [B, 1, H, W], sigmoid 到 [0, 1]
S = R * L
```

现有分析结果表明：

- `R x L` 可以较好重建低光输入；
- 但 `R` 经常偏亮，并放大低光噪声、高频纹理和压缩伪影；
- `L` 不是纯光照图，存在明显输入灰度/物体结构泄漏；
- BDDnight 中 `corr(L, I)` 接近 `0.98`，LOLv2 中约 `0.89`；
- 去掉 BDSP 后，`R/L` 尺度和 `R` 语义明显失真，因此 BDSP 不应直接移除；
- 单纯调低 `recon_weight` 或加大 `smooth_weight` 已经被实验验证风险较高。

因此，当前问题不是重建能力不足，而是分解自由度过高。

## 2. 当前损失的结构性缺陷

### 2.1 Pixel anchor 只约束均值

当前 `_pixel_anchor_loss` 的逻辑是：

```text
anchor v1: mean(L) ~= mean(maxRGB(I))
anchor v2: mean(L) ~= mean(RGB(I))
```

这对逐像素 `L [B, 1, H, W]` 来说太弱。模型只要让 `L` 的全局均值对齐即可，空间结构可以任意分配，导致：

- `L` 可以复制输入灰度结构；
- `R` 和 `L` 可以通过尺度互补完成重建；
- anchor loss 很低也不能说明光照分离正确。

### 2.2 硬重建会强迫模型解释噪声

当前重建项是：

```text
|R * L - I|
```

但真实低光图像，尤其 LOLv2 Real 和 BDDnight，包含明显传感器噪声、压缩伪影、雨夜高光点和暗部色噪。若模型只能使用乘法分解 `I = R * L`，噪声必须进入 `R` 或 `L`：

- 进入 `R`：增强后反射图噪声明显；
- 进入 `L`：光照图出现斑驳和物体结构；
- 两者混合：分解语义不稳定。

### 2.3 BDSP 保护结构，也可能保护噪声

BDSP 当前约束：

```text
|BDSP(R) - BDSP(I)|
```

它对 pure-low-single 很重要，因为它约束 `R` 不要完全偏离输入结构。但它也会把输入中的部分噪声、高频纹理和压缩边缘当作结构保留下来。因此 BDSP 应保留，但需要新增针对暗区和平坦区域的 `R` 去噪约束来制衡。

### 2.4 Retinex smooth 使用模型输出 R 作为 guide

当前 smooth 的基本形式是：

```text
|grad(L)| * exp(-alpha * |grad(R)|)
```

问题是 `R` 是模型输出，不是稳定先验。如果 `R` 中含有大量噪声或纹理，`grad(R)` 会变大，smooth 对 `L` 的约束就会在很多位置失效。换句话说，模型可以通过让 `R` 变复杂来绕开 `L` smooth。

## 3. 相关模型损失函数的可借鉴点

### 3.1 RRDNet

RRDNet 使用 zero-shot 单图优化，核心启发是：

```text
I = R * S + N
S0 = max_c(I_c)
```

其损失包含：

- 自重建；
- 光照锚点 `S ~= S0`；
- 反射约束 `R ~= I / S`；
- 光照平滑；
- 显式噪声项 `N`，允许暗区噪声被剥离。

可借鉴点：

- `L` 应该有逐像素或低频结构锚点，而不是只有均值；
- 低光噪声不应该被强迫进入 `R * L`；
- 暗区噪声需要特殊处理。

### 3.2 IGDNet

IGDNet 也使用 zero-shot 思路，分解损失包含：

```text
|I - (L * CR + N)|
+ |L - L0|
+ |CR - I / L0|
+ TV
+ noise regularization
```

可借鉴点：

- `L0 = max_c(I)` 是强光照先验；
- 增加 `R` 或 `CR` 与 `I / L0` 的一致约束，可以减少尺度漂移；
- 显式或隐式噪声建模可以避免 R 吸收暗区噪声。

### 3.3 PairLIE

PairLIE 使用同一场景下不同曝光的低光图对，依赖：

```text
R1 ~= R2
```

可借鉴点：

- pure-low-single 没有真实低光对，因此不能直接用 `R1 ~= R2`；
- 但可以引入弱反射一致约束，例如 `R ~= stopgrad(I / L)`，防止 `R/L` 相互甩锅；
- 更强的版本应放到 `pure_low_double` 中，通过双 view 做反射一致性。

### 3.4 RetinexDIP

RetinexDIP 使用 DIP 单图优化，关键损失包括：

- 重建；
- 光照一致性 `L ~= max_c(I)`；
- 反射 TV；
- 光照加权平滑。

可借鉴点：

- 对逐像素 `L`，应使用逐像素或低频逐像素的结构引导；
- `R` 需要去噪/TV 类正则，否则暗部噪声容易进入反射图。

### 3.5 DI-Retinex

DI-Retinex 不直接做传统 `R/L` 分解，而是预测亮度和对比度调整因子，并通过逆向退化损失约束无监督增强。

可借鉴点：

- 对过曝或高亮异常区域做 mask；
- 不一定要强迫所有像素同等参与重建；
- 可以用亮度相关权重减少极暗/极亮区域对 loss 的错误支配。

## 4. 推荐的新损失设计

建议把 `pure_low_single_pixel` 从当前四项扩展为：

```text
L_total =
  lambda_recon * L_recon
+ lambda_anchor_struct * L_anchor_struct
+ lambda_anchor_mean * L_anchor_mean
+ lambda_bdsp * L_bdsp
+ lambda_l_smooth * L_l_smooth_input
+ lambda_r_tv * L_r_tv_dark
+ lambda_r_consistency * L_r_consistency
+ lambda_r_sat * L_r_saturation
```

其中优先级最高的是：

1. 低频逐像素 `L` anchor；
2. input-guided `L` smooth；
3. 暗区/平坦区 `R` TV；
4. 噪声容忍重建或低频重建；
5. 弱反射一致性和 R 饱和惩罚。

## 5. 具体修改方案

### 5.1 新增 anchor_version=v3

当前 pixel anchor 只约束均值，建议新增：

```text
L0 = max_c(I)
L0_base = blur(L0)

L_anchor_struct = |L - stopgrad(L0_base)|
L_anchor_mean = |mean(L) - mean(L0)|
```

这样做不是让 `L` 精确复制 `maxRGB`，而是让 `L` 接近低频光照分布。`blur` 很关键，否则 `L` 会继续学习物体边缘和纹理。

建议实现：

```python
def _blur_illumination_map(x, kernel_size=15):
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1)
```

建议逻辑：

```python
def _pixel_anchor_loss_v3(L, I):
    l0 = I.amax(dim=1, keepdim=True)
    l0_base = _blur_illumination_map(l0, kernel_size=15).detach()
    struct = F.l1_loss(L, l0_base)
    mean = F.l1_loss(
        L.mean(dim=(2, 3), keepdim=True),
        l0.mean(dim=(2, 3), keepdim=True).detach(),
    )
    return struct + 0.2 * mean
```

注意：

- `kernel_size=15` 是初始建议，LOLv2 可尝试 15/31；
- BDDnight 分辨率更大，可尝试 31；
- `l0_base.detach()` 可避免 anchor target 参与梯度。

### 5.2 新增 smooth_version=v4

当前 smooth 使用 `R` 引导。建议新增 input-guided smooth：

```text
guide = blur(gray(I)) 或 blur(maxRGB(I))
L_smooth = |grad(L)| * exp(-alpha * |grad(guide)|)
```

推荐使用 `maxRGB(I)` 的低频图作为 guide：

```python
def _retinex_smooth_v4(L, I, alpha=10.0, blur_kernel=15):
    guide = I.amax(dim=1, keepdim=True)
    guide = _blur_illumination_map(guide, kernel_size=blur_kernel).detach()

    loss = 0.0
    if L.shape[-1] > 1:
        grad_l_x = (L[:, :, :, 1:] - L[:, :, :, :-1]).abs()
        grad_g_x = (guide[:, :, :, 1:] - guide[:, :, :, :-1]).abs()
        loss = loss + (grad_l_x * torch.exp(-alpha * grad_g_x)).mean()
    if L.shape[-2] > 1:
        grad_l_y = (L[:, :, 1:, :] - L[:, :, :-1, :]).abs()
        grad_g_y = (guide[:, :, 1:, :] - guide[:, :, :-1, :]).abs()
        loss = loss + (grad_l_y * torch.exp(-alpha * grad_g_y)).mean()
    return loss
```

关键变化：

- 不再用模型输出 `R` 作为 guide；
- 在输入低频亮度平坦处强制 `L` 平滑；
- 在真实光照或明显亮度边界处允许 `L` 跳变。

### 5.3 新增暗区/平坦区 R TV

为抑制暗部噪声进入 `R`，建议新增：

```text
w_dark = exp(-beta * gray(I))
w_flat = exp(-gamma * |grad(blur(gray(I)))|)
L_r_tv_dark = w_dark * w_flat * |grad(R)|
```

建议实现：

```python
def _reflectance_dark_tv(R, I, beta=4.0, gamma=10.0, blur_kernel=15):
    gray = 0.299 * I[:, 0:1] + 0.587 * I[:, 1:2] + 0.114 * I[:, 2:3]
    guide = _blur_illumination_map(gray, kernel_size=blur_kernel).detach()
    w_dark = torch.exp(-beta * gray.detach())

    terms = []
    if R.shape[-1] > 1:
        grad_r_x = (R[:, :, :, 1:] - R[:, :, :, :-1]).abs()
        grad_g_x = (guide[:, :, :, 1:] - guide[:, :, :, :-1]).abs()
        w_x = w_dark[:, :, :, 1:] * torch.exp(-gamma * grad_g_x)
        terms.append((grad_r_x * w_x).mean())
    if R.shape[-2] > 1:
        grad_r_y = (R[:, :, 1:, :] - R[:, :, :-1, :]).abs()
        grad_g_y = (guide[:, :, 1:, :] - guide[:, :, :-1, :]).abs()
        w_y = w_dark[:, :, 1:, :] * torch.exp(-gamma * grad_g_y)
        terms.append((grad_r_y * w_y).mean())
    return sum(terms, R.new_zeros(()))
```

这项应当较弱，否则会抹掉 `R` 中真实结构。它的作用是补 BDSP 的缺口：BDSP 保结构，`R TV dark` 抑制暗部和平坦区域噪声。

### 5.4 噪声容忍重建

不建议直接把 `recon_weight` 从 `1.0` 降到 `0.3`。已有实验显示这会破坏稳定性。更稳的做法是保留原始重建，同时加入低频重建或亮度加权重建：

```text
S = R * L
L_recon_raw = |S - I|
L_recon_low = |blur(S) - blur(I)|
L_recon = L_recon_raw + eta * L_recon_low
```

或：

```text
w = 0.2 + gray(I)
L_recon_weighted = w * |S - I|
```

推荐先采用低频重建，不直接替换原始重建：

```python
loss_recon = F.l1_loss(S, I) + 0.5 * F.l1_loss(blur(S), blur(I))
```

如果后续需要进一步处理 LOLv2 暗部噪声，再尝试亮度加权。

### 5.5 弱反射一致性

借鉴 IGDNet / PairLIE 的思想，可以加入：

```text
R_target = clamp(I / stopgrad(L), 0, 1)
L_r_consistency = |R - stopgrad(R_target)|
```

建议实现：

```python
def _reflectance_consistency(R, L, I, eps=1e-3):
    L3 = torch.cat([L, L, L], dim=1).detach().clamp_min(eps)
    target = (I / L3).clamp(0.0, 1.0).detach()
    return F.l1_loss(R, target)
```

这项不应太强，因为它和重建项相关。它的主要价值是截断 `L` 的梯度，减少 `R/L` 互相补偿。

### 5.6 R 饱和惩罚

当前 LOLv2 分析中 `R/high` 约 `1.4`，说明 `R` 偏亮。可以加入弱饱和惩罚：

```text
L_r_saturation = mean(relu(R - 0.95)^2)
```

建议权重很小，仅作为防止大面积过亮的 soft constraint。

## 6. 推荐初始权重

### 6.1 通用初始配置

```yaml
loss:
  mode: "pure_low_single_pixel"
  recon_weight: 1.0
  anchor_weight: 0.05
  anchor_mean_weight: 0.02
  bdsp_weight: 0.05
  smooth_weight: 0.1
  r_tv_weight: 0.01
  r_consistency_weight: 0.05
  r_saturation_weight: 0.005
  anchor_version: "v3"
  smooth_version: "v4"
```

### 6.2 LOLv2 建议

LOLv2 暗部噪声更明显，建议稍加强 BDSP 和 R TV：

```yaml
loss:
  recon_weight: 1.0
  anchor_weight: 0.05
  anchor_mean_weight: 0.02
  bdsp_weight: 0.08
  smooth_weight: 0.1
  r_tv_weight: 0.02
  r_consistency_weight: 0.05
  r_saturation_weight: 0.005
  anchor_version: "v3"
  smooth_version: "v4"
```

### 6.3 BDDnight 建议

BDDnight 场景结构复杂，灯光、高反射道路、雨夜纹理多，R TV 不宜过强：

```yaml
loss:
  recon_weight: 1.0
  anchor_weight: 0.05
  anchor_mean_weight: 0.02
  bdsp_weight: 0.05
  smooth_weight: 0.05
  r_tv_weight: 0.005
  r_consistency_weight: 0.03
  r_saturation_weight: 0.005
  anchor_version: "v3"
  smooth_version: "v4"
```

## 7. 推荐实现顺序

### 阶段 1：最小结构修复

只实现：

- `anchor_version=v3`
- `smooth_version=v4`

目的：

- 解决 `L` 只有均值 anchor 的根本问题；
- 避免 `R` 作为 smooth guide 导致 smooth 被模型绕开。

预期观察：

- `corr(L, I)` 应下降；
- `anchor_abs_error` 不一定可直接和 v1/v2 比较；
- `R x L` 重建不应明显恶化；
- `L` 应更像低频光照图。

### 阶段 2：加入 R 暗区去噪

新增：

- `r_tv_weight`
- `_reflectance_dark_tv`

目的：

- 抑制 LOLv2 暗区噪声和 BDDnight 平坦暗区纹理进入 `R`；
- 与 BDSP 形成互补。

预期观察：

- `R TV/input` 下降；
- `R` 暗区噪声减少；
- 不能出现真实边缘被抹平。

### 阶段 3：加入弱 R consistency 和 R saturation

新增：

- `r_consistency_weight`
- `r_saturation_weight`

目的：

- 稳定 `R/L` 尺度；
- 缓解 LOLv2 中 `R/high` 偏高的问题。

预期观察：

- `R/high` 应下降；
- `R>0.95` 比例应下降；
- `R` 不应整体变灰或过暗。

### 阶段 4：尝试噪声容忍重建

新增：

- low-frequency recon；
- 或 brightness-weighted recon。

目的：

- 允许暗部高频噪声不被完全硬重建；
- 模拟 RRDNet / IGDNet 的噪声项，但不改网络结构。

注意：

- 不建议一开始替换原始 recon；
- 应先以 additive 形式加入；
- 若 self PSNR 大幅下降，需要回退或降低该项权重。

## 8. 验证指标

不要只看 `total_loss` 或 `R x L` 的 PSNR。建议继续使用当前 pure-low-single 分析口径：

- `self_low_psnr`：确认重建没有崩；
- `r_low_tv_to_input`：观察 R 是否放大噪声/纹理；
- `l_low_input_gray_corr`：观察 L 是否复制输入灰度结构；
- `anchor_abs_error`：只在同 anchor version 内横向比较；
- `r_low_bright_095`：观察 R 是否大面积饱和；
- LOLv2 可诊断 `r_low_highref_psnr`、`r_low_highref_mean_ratio`，但 high 不能当严格 R 真值；
- BDDnight 重点看视觉图和 full-resolution recon。

视觉检查必须包含：

- input low；
- R；
- L；
- `R x L`；
- `|R x L - input|`；
- LOLv2 额外看 high reference 和 `|R - high|`。

## 9. 不建议的方向

### 9.1 不建议继续单纯加大 smooth_weight

已有实验显示 smooth 过强可能：

- 降低自重建；
- 把高频噪声推入 R；
- 对 BDDnight 雨夜纹理产生副作用。

### 9.2 不建议移除 BDSP

无 BDSP 的实验是明确负例。虽然 BDSP 可能保留部分噪声，但它对 pure-low-single 的 R/L 尺度和结构约束仍然关键。

### 9.3 不建议直接降低 recon_weight 到 0.3

已有实验中 `recon_weight=0.3` 导致稳定性和重建明显变差。更合理的是保留 `recon_weight=1.0`，再加入低频重建或亮度加权。

### 9.4 不建议直接把 L 强行拟合原始 maxRGB

`L ~= maxRGB(I)` 如果不低通，会让 L 继续复制输入纹理和物体边缘。必须使用 `blur(maxRGB(I))` 或类似低频先验。

## 10. 最小代码改动清单

建议优先修改 `loss/decomposition_loss.py`：

1. `_ANCHOR_VERSIONS` 增加 `v3`；
2. 新增 `_blur_illumination_map`；
3. `_pixel_anchor_loss` 支持 `v3`；
4. `_SMOOTH_VERSIONS` 增加 `v4`；
5. 新增 `_retinex_smooth_v4(L, I)`；
6. 修改 `PureLowSingleLoss.forward`，当 `smooth_version == "v4"` 时传入 `I`；
7. 新增可选参数：
   - `anchor_mean_weight`
   - `r_tv_weight`
   - `r_consistency_weight`
   - `r_saturation_weight`
8. 在 `_VALID_LOSS_FIELDS['pure_low_single_pixel']` 中允许这些新字段；
9. 新增对应 YAML 配置；
10. 跑 `_analysis/RetinexPixelTrans/pure_low_single/steps/99_run_all_pure_low_single_steps.py` 复用现有分析链路。

## 11. 推荐实验矩阵

先在 LOLv2 上做小矩阵：

```text
A: baseline current best
B: anchor v3 + smooth v4
C: B + r_tv=0.01
D: B + r_tv=0.02
E: C + r_consistency=0.05 + r_sat=0.005
F: E + lowfreq_recon=0.5
```

BDDnight 上更保守：

```text
A: baseline current best
B: anchor v3 + smooth v4, smooth=0.05
C: B + r_tv=0.005
D: C + r_consistency=0.03 + r_sat=0.005
```

如果阶段 1 已经明显降低 `L` 结构泄漏，而不破坏重建，应优先稳定这条线，而不是同时引入所有新项。

## 12. 总结

当前 `RetinexPixelTrans/pure_low_single` 的核心问题是“分解语义欠约束”，不是“重建 loss 不够强”。最关键的修改是让逐像素 `L` 获得合理的低频空间先验，并减少模型通过 noisy `R` 绕过 smooth 的机会。

推荐优先级：

1. `anchor_version=v3`: `L ~= blur(maxRGB(I))`；
2. `smooth_version=v4`: 使用输入低频亮度引导 L smooth；
3. `r_tv_weight`: 在暗且平坦区域抑制 R 噪声；
4. `r_consistency_weight` 和 `r_saturation_weight`: 稳定 R/L 尺度；
5. low-frequency recon 或 brightness-weighted recon: 进一步处理低光噪声。

保留原则：

- 保留 `recon_weight=1.0` 作为稳定骨架；
- 保留 BDSP；
- 不跨 anchor version 直接比较 anchor error；
- 不用 `total_loss` 跨不同权重判断实验优劣；
- LOLv2 high reference 只作为诊断，不作为严格反射率真值。
